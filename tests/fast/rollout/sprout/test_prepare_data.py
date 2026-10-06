"""prepare_data.py on synthetic rows: the driver config it writes, and Sprout's driver reading it.

``sprout-driver.json`` is all Sprout's RL driver knows of Miles's groups. A
search needs its ``miles.review`` section, without which every
``POST /rollout-groups`` with a search is an HTTP 400, and its ``api_key_env``,
without which the review job is refused once the roots are graded. The script
runs as an operator runs it (``main``, so ``prepare`` on the rows it reads),
from a local row file and parser file, with nothing downloaded. With a Sprout
checkout (``SPROUT_REPO``, else one beside this one: the lookup
``test_gsml_wiring.py`` uses) the written config also goes through Sprout's
own driver: the loader ``python -m sprout.rl_driver`` runs, then a search
request as ``SproutRolloutFn`` builds it.
"""

import asyncio
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
import pytest
from tests.fast.rollout.sprout.test_gsml_wiring import sprout_examples
from tests.fast.rollout.sprout.test_search_request import construct, search_args

from miles.utils.types import Sample

SCRIPT = Path(__file__).resolve().parents[4] / "examples" / "swe-rebench-sprout" / "prepare_data.py"
TASK_IDS = ("octo__widgets-1", "octo__widgets-2")
#: Every review setting away from its default, and the shortest grade wall time Sprout's driver takes.
EVERY_FLAG = {
    "--review-profile": "review-qwen",
    "--review-max-output-tokens": "24576",
    "--review-prompt-chars": "200000",
    "--review-temperature": "0.6",
    "--review-reasoning-effort": "medium",
    "--review-max-concurrent-requests": "4",
    "--grade-wall-seconds": "240",
}


@pytest.fixture(scope="module")
def prepare_data() -> ModuleType:
    spec = importlib.util.spec_from_file_location("swe_rebench_sprout_prepare_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(instance_id: str) -> dict:
    """A task row with the fields prepare_data.py reads, spelled as SWE-rebench-V2 spells them."""
    return {
        "instance_id": instance_id,
        "repo": "octo/widgets",
        "base_commit": "0" * 40,
        "problem_statement": "Widget.resize drops the border.",
        "test_patch": "diff --git a/tests/test_widget.py b/tests/test_widget.py\n",
        "FAIL_TO_PASS": ["tests/test_widget.py::test_resize_keeps_the_border"],
        "PASS_TO_PASS": ["tests/test_widget.py::test_resize"],
        "install_config": {"test_cmd": "pytest -rA"},
        "image_name": f"docker.io/swerebenchv2/octo-widgets:{instance_id}",
    }


def run_script(prepare_data: ModuleType, monkeypatch, tmp_path: Path, flags: dict[str, str]) -> Path:
    """``prepare_data.py`` over the synthetic rows with ``flags`` beside the required ones; its output directory."""
    rows = tmp_path / "rows.jsonl"
    rows.write_text("".join(json.dumps(row(task_id)) + "\n" for task_id in TASK_IDS))
    parser_file = tmp_path / "log_parsers.py"
    parser_file.write_text("def parse_log_pytest(log, test_spec):\n    return {}\n")
    argv = {
        "--output-dir": str(tmp_path / "out"),
        "--sprout-data-dir": str(tmp_path / "sprout-host"),
        "--model": "Qwen/Qwen3.8-27B",
        "--source-jsonl": str(rows),
        "--parser-file": str(parser_file),
        **flags,
    }
    monkeypatch.setattr(sys, "argv", ["prepare_data.py", *(item for pair in argv.items() for item in pair)])
    prepare_data.main()
    return tmp_path / "out"


def driver_config(output: Path) -> dict:
    return json.loads((output / "sprout-driver.json").read_text())


def test_the_driver_config_carries_what_a_search_needs(prepare_data, monkeypatch, tmp_path):
    miles = driver_config(run_script(prepare_data, monkeypatch, tmp_path, {}))["miles"]
    assert miles["review"] == {
        "profile": "review",
        "max_output_tokens": 32768,
        "prompt_chars": 250000,
        "temperature": 0.8,
    }, "and nothing else: Sprout then sends no reasoning_effort and runs up to 8 of a review's calls at once"
    assert miles["grade_wall_seconds"] == 600
    assert miles["api_key_env"] == "SPROUT_MODEL_API_KEY", "what the example mini and review profiles forward"
    assert set(miles) == {
        "profile",
        "run_defaults",
        "image_resources",
        "grade_wall_seconds",
        "review",
        "tasks",
        "api_key_env",
    }
    assert sorted(miles["tasks"]) == list(TASK_IDS)


def test_every_review_setting_and_the_grade_wall_time_are_taken_from_their_flags(prepare_data, monkeypatch, tmp_path):
    miles = driver_config(run_script(prepare_data, monkeypatch, tmp_path, EVERY_FLAG))["miles"]
    assert miles["review"] == {
        "profile": "review-qwen",
        "max_output_tokens": 24576,
        "prompt_chars": 200000,
        "temperature": 0.6,
        "reasoning_effort": "medium",
        "max_concurrent_requests": 4,
    }
    assert miles["grade_wall_seconds"] == 240


@pytest.mark.parametrize(
    "flag, value",
    [
        ("--grade-wall-seconds", "239"),
        ("--review-profile", ""),
        ("--review-reasoning-effort", ""),
        ("--review-temperature", "2.5"),
        ("--review-max-output-tokens", "0"),
        ("--review-prompt-chars", "0"),
        ("--review-max-concurrent-requests", "0"),
    ],
)
def test_a_setting_sprout_cannot_run_with_is_refused_before_anything_is_written(
    prepare_data, monkeypatch, tmp_path, flag, value
):
    with pytest.raises(SystemExit):
        run_script(prepare_data, monkeypatch, tmp_path, {flag: value})
    assert not (tmp_path / "out").exists()


@pytest.fixture
def sprout(monkeypatch) -> SimpleNamespace:
    """Sprout's RL driver modules, imported from the checkout ``sprout_examples`` finds; skipped without one."""
    directory = sprout_examples()
    if directory is None:
        pytest.skip("no Sprout checkout beside this one: set SPROUT_REPO to run the config through Sprout's driver")
    source = directory.parents[1].resolve()  # the checkout's src/
    monkeypatch.syspath_prepend(str(source))
    modules = SimpleNamespace(
        messages=importlib.import_module("sprout.rl_driver.messages"),
        ledger=importlib.import_module("sprout.rl_driver.ledger"),
        loader=importlib.import_module("sprout.rl_driver.__main__"),
    )
    assert Path(modules.messages.__file__).resolve().is_relative_to(source), "a Sprout other than the checkout's"
    return modules


def serve(sprout: SimpleNamespace, config_path: Path, monkeypatch, capsys):
    """``python -m sprout.rl_driver --config <config_path>`` up to the server it starts: the app it would serve."""
    served = []
    monkeypatch.setattr(sprout.loader.uvicorn, "run", lambda app, **address: served.append(app))
    monkeypatch.setattr(sys, "argv", ["sprout.rl_driver", "--config", str(config_path)])
    monkeypatch.setenv(json.loads(config_path.read_text())["runstore_token_env"], "runstore-token")
    try:
        sprout.loader.main()
    except SystemExit:
        pytest.fail(f"Sprout's RL driver refused the config: {capsys.readouterr().err}")
    (app,) = served
    return app


async def post(app, path: str, body: dict) -> httpx.Response:
    """One request to the app, without starting it: no lifespan, so no polling of a Run Store."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://driver") as client:
        return await client.post(path, json=body)


@pytest.mark.parametrize("flags", [{}, EVERY_FLAG], ids=["defaults", "every-flag"])
def test_sprouts_driver_reads_the_config_and_takes_a_search_request_as_miles_builds_it(
    prepare_data, sprout, monkeypatch, tmp_path, capsys, flags
):
    output = run_script(prepare_data, monkeypatch, tmp_path, flags)
    config = driver_config(output)
    review = config["miles"]["review"]
    assert set(review) <= sprout.messages.REVIEW_CONFIG_FIELDS
    app = serve(sprout, output / "sprout-driver.json", monkeypatch, capsys)

    # The first task's prompt group as the data source hands it out, and the search request Miles submits for it.
    first = json.loads((output / "miles.jsonl").read_text().splitlines()[0])
    group = [
        Sample(prompt=first["prompt"], index=index, group_index=0, metadata=first["metadata"]) for index in (0, 1)
    ]
    request, _ = construct(search_args())._build_request(group=group, rollout_id=1)
    assert request.search is not None
    response = asyncio.run(post(app, "/rollout-groups", request.wire()))
    assert response.status_code == 202, response.text

    # The review job the driver submits once the roots are graded, for either review the outcome may call for.
    ledger = sprout.ledger.Ledger(config["ledger"])
    document = ledger.get(sprout.messages.internal_id(request.rollout_job_id))["document"]
    settings = {key: value for key, value in review.items() if key != "profile"}
    templates = document["search"]["review_templates"]
    assert templates, "the driver keeps a review job template for each review the outcome may call for"
    for template in templates.values():
        assert template["profile"] == review["profile"]
        assert {key: template["spec"][key] for key in settings} == settings
        assert template["spec"]["api_key_env"] == "SPROUT_MODEL_API_KEY", "a review job without one is refused"
    assert document["grade_wall_seconds"] == config["miles"]["grade_wall_seconds"]
