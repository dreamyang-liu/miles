#!/usr/bin/env python3
"""Append a \\boxed{} answer-format instruction to each prompt in a JSONL dataset.

Why: the calculator multi-turn reward (calculator_tool.reward_func) grades the
final answer most reliably when the model wraps it in \\boxed{}, which routes
through the deepscaler mathd/sympy grader instead of a brittle numeric-regex
fallback. This script makes the requirement explicit in the prompt.

Usage:
    python make_boxed_prompts.py training_prompts.jsonl training_prompts_boxed.jsonl
"""
import json
import sys

SUFFIX = "Put your final numerical answer inside \\boxed{} (for example, \\boxed{42})."


def main(src: str, dst: str) -> None:
    n = changed = 0
    with open(src) as f, open(dst, "w") as g:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            n += 1
            content = d["prompt"][0]["content"]
            if "\\boxed" not in content:
                d["prompt"][0]["content"] = content.rstrip() + "\n" + SUFFIX
                changed += 1
            g.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Processed {n} prompts, appended \\boxed instruction to {changed} -> {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
