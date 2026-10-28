"""RafoGenerate through miles' argument parsing and a mock SGLang server."""

import json
import re

import pytest
from tests.fast.examples.experimental.rafo.forest_fixtures import TASK, WRONG, make_task_sample
from tests.fast.fixtures.generation_fixtures import MODEL_NAME, compute_generate_args, make_args

from miles.ray.rollout.train_data_conversion import convert_samples_to_train_data
from miles.rollout.base_types import GenerateFnInput
from miles.rollout.inference_rollout.compatibility import load_generate_function
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState
from miles.utils.misc import SingletonMeta
from miles.utils.test_utils.mock_sglang_server import ProcessResult, with_mock_server

GENERATE_PATH = "examples.experimental.rafo.generate.RafoGenerate"
SAMPLING_PARAMS = {"max_new_tokens": 64, "temperature": 1.0}
RAFO_ARGV = [
    "--rafo-env-path",
    "examples.experimental.rafo.toy_env.make_toy_env",
    "--rafo-tool-specs-path",
    "examples.experimental.rafo.toy_env.TOOL_SPECS",
    "--rafo-tool-call-parser",
    "qwen25",
    "--rafo-m",
    "2",
    "--disable-rewards-normalization",
]


def _agent_turn(prompt: str) -> ProcessResult:
    """The mock policy: patch the file that the latest check found suspicious, otherwise a wrong file."""
    hits = re.findall(r"Inspected (\w+\.py): found a suspicious line", prompt)
    call = {"name": "patch", "arguments": {"path": hits[-1] if hits else WRONG}}
    return ProcessResult(text=f"Let me fix it.\n<tool_call>\n{json.dumps(call)}\n</tool_call><|im_end|>")


@pytest.fixture
def rafo_env(request):
    SingletonMeta.clear_all_instances()
    extra = getattr(request, "param", [])
    with with_mock_server(model_name=MODEL_NAME, process_fn=_agent_turn) as server:
        args = make_args(
            variant="single_turn",
            router_port=server.port,
            custom_generate_function_path=GENERATE_PATH,
            extra_argv=RAFO_ARGV + extra,
        )
        yield compute_generate_args(args), server
    SingletonMeta.clear_all_instances()


async def _generate(args, *, evaluation: bool = False):
    state = GenerateState(args)
    sample = make_task_sample()
    output = await load_generate_function(GENERATE_PATH)(
        GenerateFnInput(state=state, sample=sample, sampling_params=dict(SAMPLING_PARAMS), evaluation=evaluation)
    )
    return sample, output.samples, state


class TestRafoGenerate:
    async def test_one_call_returns_the_training_samples_of_one_task_forest(self, rafo_env) -> None:
        args, server = rafo_env
        task, samples, state = await _generate(args)

        # both roots fail; each review exposes the bug and both continuations then succeed: the groups have one
        # outcome each, so only the two reviewer turns carry a signal
        assert [s.train_metadata["rafo_kind"] for s in samples] == ["reflect", "reflect"]
        assert [s.train_metadata["rafo_coef"] for s in samples] == [pytest.approx(1 / (4 * 32))] * 2
        assert all(s.rollout_id == task.index and s.metadata["raw_reward"] == 0.0 for s in samples)
        for sample in samples:
            assert len(sample.loss_mask) == len(sample.rollout_log_probs) == sample.response_length
            text = state.tokenizer.decode(sample.tokens[-sample.response_length :])
            assert f"I should check {TASK.bug}" in text and text.endswith("<|im_end|>")
        assert len(server.request_log) == 2 * TASK.max_patches + 2 * 2  # root turns + one turn per continuation
        reviewed = [
            p for p in server.request_log if f"I should check {TASK.bug}" in state.tokenizer.decode(p["input_ids"])
        ]
        assert len(reviewed) == 4 and all(p["sampling_params"]["max_new_tokens"] == 64 for p in reviewed)

    async def test_evaluation_runs_the_agent_alone(self, rafo_env) -> None:
        args, server = rafo_env
        _, sample, _ = await _generate(args, evaluation=True)

        assert sample.reward == 0.0 and sample.train_metadata is None
        assert len(server.request_log) == TASK.max_patches

    @pytest.mark.parametrize("rafo_env", [["--n-samples-per-prompt", "2"]], indirect=True)
    async def test_rollout_settings_that_break_the_forest_are_rejected(self, rafo_env) -> None:
        args, _ = rafo_env
        with pytest.raises(ValueError, match="n-samples-per-prompt"):
            await _generate(args)

    def test_forest_settings_without_an_rloo_group_are_rejected_at_launch(self) -> None:
        with pytest.raises(ValueError, match="rafo-k"):
            make_args(
                variant="single_turn",
                router_port=1,
                custom_generate_function_path=GENERATE_PATH,
                extra_argv=[*RAFO_ARGV, "--rafo-k", "1"],
            )


class TestTrainDataContract:
    async def test_forest_samples_become_train_rows_with_their_rafo_roles_and_one_rollout_per_task(
        self, rafo_env
    ) -> None:
        args, _ = rafo_env
        state = GenerateState(args)
        fn = load_generate_function(GENERATE_PATH)
        samples = []
        for index in (3, 4):
            task = make_task_sample()
            task.index = task.group_index = index
            output = await fn(
                GenerateFnInput(state=state, sample=task, sampling_params=dict(SAMPLING_PARAMS), evaluation=False)
            )
            samples += output.samples

        data = convert_samples_to_train_data(args, samples, {}, None, None)

        assert data["metadata"] == [s.train_metadata for s in samples]
        assert data["rollout_ids"] == [3, 3, 4, 4]
        assert data["raw_reward"] == [0.0] * 4
        assert [sum(mask) for mask in data["loss_masks"]] == [sum(s.loss_mask) for s in samples]
