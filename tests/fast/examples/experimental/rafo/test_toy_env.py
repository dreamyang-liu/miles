import json

from examples.experimental.rafo.protocols import ReviewRequest, ToolCall
from examples.experimental.rafo.toy_env import ToyEnv, ToyTask, toy_reviewer, write_toy_dataset
from miles.utils.types import Sample


def _call(name: str, path: str) -> ToolCall:
    return ToolCall(name=name, arguments={"path": path}, id="c")


class TestToyEnv:
    async def test_a_wrong_patch_is_a_failed_check_that_shows_the_failing_line(self) -> None:
        task = ToyTask(files=("a.py", "b.py"), ranges=((1, 9), (10, 20)), bug="b.py", bug_line=12, max_patches=2)
        env = ToyEnv(task=task)

        step = await env.step([_call("patch", "a.py")], check_only=False)

        assert step.failed_check and not step.done and "line 12" in step.outputs[0]
        assert (await env.step([_call("patch", "b.py")], check_only=False)).done and await env.reward() == 1.0

    async def test_a_check_cannot_change_the_code(self) -> None:
        task = ToyTask(files=("a.py", "b.py"), ranges=((1, 9), (10, 20)), bug="b.py", bug_line=12)
        env = ToyEnv(task=task)

        step = await env.step([_call("inspect", "b.py"), _call("patch", "b.py")], check_only=True)

        assert "suspicious" in step.outputs[0] and "Refused" in step.outputs[1]
        assert env.patches == 0 and await env.reward() == 0.0

    async def test_forks_are_independent(self) -> None:
        task = ToyTask(files=("a.py", "b.py"), ranges=((1, 9), (10, 20)), bug="b.py", bug_line=12)
        env = ToyEnv(task=task)
        fork = await env.fork()

        await fork.step([_call("patch", "b.py")], check_only=False)

        assert await fork.reward() == 1.0 and await env.reward() == 0.0 and env.opened == [env, fork]


class TestToyDataset:
    def test_rows_hold_a_chat_prompt_and_a_label_whose_bug_line_is_in_the_bug_file(self, tmp_path) -> None:
        path = tmp_path / "train.jsonl"
        write_toy_dataset(str(path), n_tasks=20, seed=0)

        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 20
        for row in rows:
            task = ToyTask.from_label(row["label"])
            lo, hi = task.ranges[task.files.index(task.bug)]
            assert lo <= task.bug_line <= hi
            assert row["prompt"] == [{"role": "user", "content": task.prompt()}]

    def test_the_dataset_is_reproducible(self, tmp_path) -> None:
        write_toy_dataset(str(tmp_path / "a.jsonl"), n_tasks=5, seed=1)
        write_toy_dataset(str(tmp_path / "b.jsonl"), n_tasks=5, seed=1)

        assert (tmp_path / "a.jsonl").read_text() == (tmp_path / "b.jsonl").read_text()


async def test_the_oracle_reviewer_exposes_the_bug_with_a_check_and_no_fix() -> None:
    task = ToyTask(files=("a.py", "b.py"), ranges=((1, 9), (10, 20)), bug="b.py", bug_line=12)
    request = ReviewRequest(sample=Sample(label=task.to_label()), messages=(), checkpoint=0, depth=1, tools=())

    turn = await toy_reviewer(request)

    assert [(c.name, c.arguments) for c in turn.tool_calls] == [("inspect", {"path": "b.py"})]
    assert "line 12" in turn.reasoning and "patch b.py" not in turn.reasoning
