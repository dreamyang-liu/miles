"""
Multi-turn calculator tool generate function + reward for agentic RL training.

Usage in Miles:
  --custom-generate-function-path calculator_tool.generate
  --custom-rm-path calculator_tool.reward_func
"""

import json
import math
import re

from miles.rollout.sglang_rollout import GenerateState
from miles.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward
from miles.rollout.rm_hub.math_utils import grade_answer_mathd, grade_answer_sympy
from miles.utils.http_utils import post
from miles.utils.types import Sample

MAX_TURNS = 16

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "A calculator tool that evaluates mathematical expressions. "
            "Supports basic arithmetic (+, -, *, /), exponentiation (**), modulo (%), "
            "and common math functions (sqrt, abs, round, min, max, sum, pow).",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate, e.g. '(48 + 24)' or 'sqrt(144)'",
                    }
                },
                "required": ["expression"],
            },
        },
    }
]

_SAFE_GLOBALS = {
    "__builtins__": {},
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "int": int,
    "float": float,
    "sqrt": math.sqrt,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "log10": math.log10,
    "pi": math.pi,
    "e": math.e,
}


def execute_calculator(expression: str) -> str:
    if not expression.strip():
        return "Error: empty expression"
    try:
        result = eval(expression, _SAFE_GLOBALS, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def parse_tool_calls(text: str):
    """Parse tool calls from assistant output.

    Supports two formats:
    1. JSON: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    2. XML:  <tool_call><function=name><parameter=key>value</parameter></function></tool_call>
    """
    blocks = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    results = []
    for block in blocks:
        block = block.strip()
        # Try JSON format first
        if block.startswith('{'):
            try:
                results.append(json.loads(block))
                continue
            except json.JSONDecodeError:
                pass
        # Try XML format: <function=name><parameter=key>value</parameter></function>
        func_match = re.search(r'<function=([^>]+)>(.*?)</function>', block, re.DOTALL)
        if func_match:
            name = func_match.group(1).strip()
            params_text = func_match.group(2)
            arguments = {}
            for param_match in re.finditer(r'<parameter=([^>]+)>(.*?)</parameter>', params_text, re.DOTALL):
                arguments[param_match.group(1).strip()] = param_match.group(2).strip()
            results.append({"name": name, "arguments": arguments})
    return results


async def generate(args, sample: Sample, sampling_params) -> Sample:
    """Multi-turn generate with calculator tool calls."""
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    prompt_tokens_ids = state.tokenizer(sample.prompt, add_special_tokens=False)["input_ids"]
    response_token_ids = []
    loss_masks = []
    response = ""
    tool_call_count = 0

    for _turn in range(MAX_TURNS):
        total_length = len(prompt_tokens_ids) + len(response_token_ids)
        if args.rollout_max_context_len is not None:
            max_ctx = args.rollout_max_context_len
        else:
            max_ctx = args.context_parallel_size * args.max_tokens_per_gpu
        if total_length >= max_ctx:
            sample.status = Sample.Status.TRUNCATED
            break

        current_token_ids = prompt_tokens_ids + response_token_ids
        payload = {
            "input_ids": current_token_ids,
            "sampling_params": sampling_params,
            "return_logprob": True,
        }

        output = await post(url, payload)

        if output["meta_info"]["finish_reason"]["type"] == "abort":
            sample.status = Sample.Status.ABORTED
            return sample

        if "output_token_logprobs" in output["meta_info"]:
            cur_token_ids = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
            cur_response = state.tokenizer.decode(cur_token_ids)
            cur_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
            if sample.rollout_log_probs is None:
                sample.rollout_log_probs = []
            sample.rollout_log_probs += cur_log_probs
        else:
            cur_response = output["text"]
            cur_token_ids = state.tokenizer(cur_response, add_special_tokens=False)["input_ids"]

        response += cur_response
        response_token_ids += cur_token_ids
        loss_masks += [1] * len(cur_token_ids)

        if output["meta_info"]["finish_reason"]["type"] == "length":
            sample.status = Sample.Status.TRUNCATED
            break

        tool_calls = parse_tool_calls(cur_response)
        if not tool_calls:
            break

        tool_results = []
        for call in tool_calls:
            name = call.get("name", "")
            arguments = call.get("arguments", {})
            if name == "calculator":
                expr = arguments.get("expression", "")
                result = execute_calculator(expr)
            else:
                result = f"Error: unknown tool '{name}'"
            tool_results.append(result)
            tool_call_count += 1

        obs = "\n<tool_response>\n" + "\n".join(tool_results) + "\n</tool_response>\n"
        obs_token_ids = state.tokenizer(obs, add_special_tokens=False)["input_ids"]
        response += obs
        response_token_ids += obs_token_ids
        loss_masks += [0] * len(obs_token_ids)

        if sample.rollout_log_probs is not None:
            sample.rollout_log_probs += [0.0] * len(obs_token_ids)

    sample.tokens = prompt_tokens_ids + response_token_ids
    sample.response_length = len(response_token_ids)
    sample.response = response
    sample.loss_mask = loss_masks

    if not hasattr(sample, 'status') or sample.status is None:
        match output["meta_info"]["finish_reason"]["type"]:
            case "length":
                sample.status = Sample.Status.TRUNCATED
            case "abort":
                sample.status = Sample.Status.ABORTED
            case "stop":
                sample.status = Sample.Status.COMPLETED

    return sample


def _candidate_answers(final_text: str) -> list[str]:
    """Extract candidate numeric answers from the final assistant text, most
    likely first. Robust to thousands separators, trailing units/punctuation,
    and answers that are not on the very last line.

    Order of preference:
      1. \\boxed{...} content
      2. numbers following an explicit answer cue ("answer is X", "= X", "is X")
      3. any number in the final text (last one first)
    """
    # Normalize thousands separators so "1,000" -> "1000" (avoid splitting into "000").
    text = re.sub(r'(?<=\d),(?=\d{3}\b)', '', final_text)

    candidates: list[str] = []

    def _push(num: str):
        num = num.rstrip('.')  # strip trailing sentence period, keep decimals like 5.0
        if num and num not in candidates:
            candidates.append(num)

    num_re = r'[-+]?\d+(?:\.\d+)?'

    # 1. \boxed{...}
    for m in re.findall(r'\\boxed\{([^}]*)\}', text):
        for n in re.findall(num_re, m):
            _push(n)

    # 2. explicit answer cues (search whole final text, last match is usually the conclusion)
    cue_re = re.compile(r'(?:answer\s*(?:is|:|=)|final answer\s*(?:is|:|=)?|=)\s*\$?\s*(' + num_re + r')', re.IGNORECASE)
    cue_matches = cue_re.findall(text)
    for n in reversed(cue_matches):
        _push(n)

    # 3. fallback: any number in the final text, last occurrence first
    all_nums = re.findall(num_re, text)
    for n in reversed(all_nums):
        _push(n)

    return candidates


async def reward_func(args, sample: Sample, **kwargs):
    """Reward = deepscaler math correctness, with fallback for tool-calling format."""
    ground_truth = sample.label if sample.label is not None else ""
    result = get_deepscaler_rule_based_reward(sample.response, ground_truth)
    if result == 0 and ground_truth:
        # Fallback: for multi-turn tool responses without \boxed{}, check if the
        # final assistant text (after last </tool_response>) contains the answer.
        if "</tool_response>" in sample.response:
            final_text = sample.response.rsplit("</tool_response>", 1)[-1]
        elif "</think>" in sample.response:
            final_text = sample.response.rsplit("</think>", 1)[-1]
        else:
            final_text = sample.response

        gt = str(ground_truth)
        for model_answer in _candidate_answers(final_text):
            if grade_answer_mathd(model_answer, gt) or grade_answer_sympy(model_answer, gt):
                result = 1
                break
    return result
