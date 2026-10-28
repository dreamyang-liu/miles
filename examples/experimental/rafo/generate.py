"""miles generate function for RAFO.

``--custom-generate-function-path examples.experimental.rafo.generate.RafoGenerate``. One call builds the forest
of one task and returns all of its training samples with one shared rollout id. With ``evaluation`` it runs one
episode of the agent alone.
"""

from argparse import Namespace
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from examples.experimental.rafo.config import RafoConfig
from examples.experimental.rafo.credit import FixedCoefficients
from examples.experimental.rafo.episode import (
    EpisodeContext,
    TurnGenerator,
    TurnResult,
    make_tool_call_parser,
    render_reviewer_turn,
    template_renders_reasoning,
)
from examples.experimental.rafo.forest import ForestContext, build_forest, run_root
from examples.experimental.rafo.protocols import EnvFactory, Reviewer, ToolCall
from examples.experimental.rafo.samples import forest_samples
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_prompt_ids_from_sample,
    compute_request_payload,
    compute_routing_headers,
    update_sample_from_response,
)
from miles.utils.function_registry import load_function
from miles.utils.http_utils import post
from miles.utils.types import Sample


@dataclass(frozen=True)
class _Setup:
    """Values derived once from the launch arguments and the tokenizer."""

    config: RafoConfig
    coefficients: FixedCoefficients
    tools: tuple[dict[str, Any], ...]
    parse_tool_calls: Callable[[str], list[ToolCall]]
    make_env: EnvFactory
    review: Reviewer
    end_of_turn_id: int
    reasoning_block: bool
    url: str


class RafoGenerate:
    config_class = RafoConfig

    def __init__(self) -> None:
        self._setup: _Setup | None = None

    async def __call__(self, input: GenerateFnInput) -> GenerateFnOutput:
        if self._setup is None:
            self._setup = _create_setup(input.args, input.state.tokenizer)
        setup = self._setup
        task = input.sample
        prompt_ids = compute_prompt_ids_from_sample(input.state, task, tools=list(setup.tools))
        ctx = _forest_context(setup, input)
        if input.evaluation:
            root = await run_root(task, prompt_ids, ctx)
            root.sample.reward = root.reward
            return GenerateFnOutput(samples=root.sample)
        forest = await build_forest(task, prompt_ids, ctx)
        if forest.aborted:
            aborted = deepcopy(task)
            aborted.status = Sample.Status.ABORTED
            return GenerateFnOutput(samples=[aborted])
        samples = forest_samples(task, forest, setup.coefficients, credit=setup.config.rafo_credit)
        return GenerateFnOutput(samples=samples)


def _create_setup(args: Namespace, tokenizer) -> _Setup:
    _check_rollout_args(args)
    values = {name: value for name, value in dict(args).items() if name in RafoConfig.model_fields}
    config = RafoConfig.model_validate(values)
    tools = tuple(load_function(config.rafo_tool_specs_path))
    end_of_turn = config.rafo_end_of_turn_token or tokenizer.eos_token
    end_of_turn_id = tokenizer.convert_tokens_to_ids(end_of_turn)
    if end_of_turn_id is None or end_of_turn_id == tokenizer.unk_token_id:
        raise ValueError(f"the end-of-turn token {end_of_turn!r} is not a token of the tokenizer")
    return _Setup(
        config=config,
        coefficients=config.coefficients(),
        tools=tools,
        parse_tool_calls=make_tool_call_parser(tools, config.rafo_tool_call_parser),
        make_env=load_function(config.rafo_env_path),
        review=load_function(config.rafo_reviewer_path),
        end_of_turn_id=end_of_turn_id,
        reasoning_block=template_renders_reasoning(tokenizer),
        url=f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate",
    )


def _check_rollout_args(args: Namespace) -> None:
    unsupported = {
        "--n-samples-per-prompt other than 1 (RAFO samples --rafo-m roots per task itself)": (
            args.n_samples_per_prompt != 1
        ),
        "--partial-rollout": args.partial_rollout,
        "--group-rm (RAFO rewards come from the environment)": args.group_rm,
        "rewards normalization (pass --disable-rewards-normalization; the samples of one task have different rewards)": (
            args.rewards_normalization
        ),
        "--use-rollout-routing-replay": args.use_rollout_routing_replay,
        "--use-rollout-indexer-replay": args.use_rollout_indexer_replay,
        "--use-output-store": args.use_output_store,
        "--rollout-top-logprobs-num": bool(args.rollout_top_logprobs_num),
        "--use-sampling-support-replay": args.use_sampling_support_replay,
    }
    if enabled := [name for name, on in unsupported.items() if on]:
        raise ValueError(f"the RAFO generate function does not support {'; '.join(enabled)}")


def _forest_context(setup: _Setup, input: GenerateFnInput) -> ForestContext:
    tokenizer = input.state.tokenizer
    episode = EpisodeContext(
        generate_turn=_sglang_turn_generator(input, setup.url),
        parse_tool_calls=setup.parse_tool_calls,
        tokenizer=tokenizer,
        max_turns=setup.config.rafo_max_turns,
    )

    def render_turn(turn) -> list[int]:
        return render_reviewer_turn(
            tokenizer,
            turn,
            tools=setup.tools,
            end_of_turn_id=setup.end_of_turn_id,
            reasoning_block=setup.reasoning_block,
        )

    return ForestContext(
        episode=episode,
        make_env=setup.make_env,
        review=setup.review,
        render_turn=render_turn,
        tools=setup.tools,
        config=setup.config.forest(),
    )


def _sglang_turn_generator(input: GenerateFnInput, url: str) -> TurnGenerator:
    args, state, evaluation = input.args, input.state, input.evaluation
    sampling_params = input.sampling_params

    async def generate_turn(sample: Sample) -> TurnResult:
        if state.aborted:
            return TurnResult(text="", finish="abort")
        payload, halt_status = compute_request_payload(
            args, sample.tokens, sampling_params, evaluation=evaluation, kv_cache_namespace=sample.kv_cache_namespace
        )
        if payload is None:
            sample.status = halt_status
            return TurnResult(text="", finish="context")
        output = await post(url, payload, headers=compute_routing_headers(args, sample))
        await update_sample_from_response(args, sample, payload=payload, output=output, update_loss_mask=True)
        return TurnResult(text=output["text"], finish=output["meta_info"]["finish_reason"]["type"])

    return generate_turn
