"""Launch arguments of the RAFO generate function; registered by ``RafoGenerate.config_class``."""

from typing import Literal

from examples.experimental.rafo.credit import FixedCoefficients, fixed_coefficients
from examples.experimental.rafo.forest import ForestConfig
from pydantic import model_validator

from miles.utils.args.schema import A, Arg, BaseConfig


class RafoConfig(BaseConfig):
    rafo_env_path: A[
        str, Arg(required=True, help="Async factory `make_env(sample) -> RafoEnv` that opens the task environment.")
    ]
    rafo_reviewer_path: A[
        str, Arg(required=True, help="Async `review(ReviewRequest) -> ReviewerTurn | None`; None means abstain.")
    ]
    rafo_tool_specs_path: A[str, Arg(required=True, help="The tool specs of the agent (a list of OpenAI tools).")]
    rafo_tool_call_parser: A[
        str, Arg(required=True, help="SGLang function-call parser that reads the agent's tool calls, e.g. qwen25.")
    ]
    rafo_m: A[int, Arg(help="Roots for each task (M).")] = 4
    rafo_k: A[int, Arg(help="Continuations for each event (K).")] = 2
    rafo_s: A[
        int, Arg(help="Slots for each task (S): at most S events; also the fixed repair and CE denominator.")
    ] = 4
    rafo_d: A[int, Arg(help="Maximum review depth (D).")] = 3
    rafo_c: A[float, Arg(help="Fixed normalizer of agent tokens (C).")] = 64.0
    rafo_c_h: A[float, Arg(help="Fixed normalizer of reviewer-turn tokens (C_h).")] = 32.0
    rafo_rereview: A[
        Literal["branch", "checkpoint"],
        Arg(
            type_parser=str,
            choices=("branch", "checkpoint"),
            help="Where a deeper review starts: inside the first failed continuation after its own first failed "
            "check (branch), or at the same checkpoint (checkpoint).",
        ),
    ] = "branch"
    rafo_credit: A[
        Literal["subtree", "own"],
        Arg(
            type_parser=str,
            choices=("subtree", "own"),
            help="Reviewer-turn weight: a continuation that a deeper review rescued counts with that review's weight "
            "(subtree), or only the event's own continuation rewards count (own).",
        ),
    ] = "subtree"
    rafo_max_turns: A[int, Arg(help="Assistant turns in one sequence, reviewer turns included.")] = 16
    rafo_end_of_turn_token: A[
        str | None, Arg(help="The token that ends an assistant turn; default: the tokenizer's eos token.")
    ] = None

    @model_validator(mode="after")
    def _check(self) -> "RafoConfig":
        if self.rafo_m < 2 or self.rafo_k < 2:
            raise ValueError("RAFO needs --rafo-m >= 2 and --rafo-k >= 2: the local RLOO groups need two samples")
        if self.rafo_s < 1 or self.rafo_d < 1 or self.rafo_max_turns < 1:
            raise ValueError("--rafo-s, --rafo-d and --rafo-max-turns must be at least 1")
        if self.rafo_c <= 0 or self.rafo_c_h <= 0:
            raise ValueError("--rafo-c and --rafo-c-h must be positive")
        return self

    def forest(self) -> ForestConfig:
        return ForestConfig(m=self.rafo_m, k=self.rafo_k, s=self.rafo_s, d=self.rafo_d, rereview=self.rafo_rereview)

    def coefficients(self) -> FixedCoefficients:
        return fixed_coefficients(m=self.rafo_m, k=self.rafo_k, s=self.rafo_s, c=self.rafo_c, c_h=self.rafo_c_h)
