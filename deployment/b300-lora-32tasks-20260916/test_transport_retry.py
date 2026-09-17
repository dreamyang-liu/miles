"""A transient result-read failure must not cancel the other task groups."""

import importlib.util
from pathlib import Path

import httpx
import pytest


spec = importlib.util.spec_from_file_location("deployment_rollout", Path(__file__).with_name("night_rollout.py"))
rollout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rollout)


def status_error(status):
    request = httpx.Request("GET", "http://ash/rollout-groups/same-id")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError("test status", request=request, response=response)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [status_error(500), status_error(503), httpx.ReadError("disconnected")])
async def test_transient_failure_retries_identical_request(monkeypatch, error):
    seen = []

    async def call(identifier):
        seen.append(identifier)
        if len(seen) < 3:
            raise error
        return {"status": "running"}

    async def sleep(_):
        pass

    monkeypatch.setattr(rollout.asyncio, "sleep", sleep)
    result = await rollout.RecordingClient._retry_transport(object(), call, "same-id")
    assert result == {"status": "running"}
    assert seen == ["same-id"] * 3


@pytest.mark.asyncio
async def test_invalid_request_is_not_retried():
    seen = []

    async def call(identifier):
        seen.append(identifier)
        raise status_error(422)

    with pytest.raises(httpx.HTTPStatusError):
        await rollout.RecordingClient._retry_transport(object(), call, "same-id")
    assert seen == ["same-id"]


@pytest.mark.asyncio
async def test_outage_has_a_bounded_pickleable_failure(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(rollout.time, "monotonic", lambda: now[0])

    async def call(identifier):
        raise status_error(500)

    async def sleep(delay):
        now[0] += delay

    monkeypatch.setattr(rollout.asyncio, "sleep", sleep)
    with pytest.raises(RuntimeError, match="300 seconds"):
        await rollout.RecordingClient._retry_transport(object(), call, "same-id")
    assert now[0] == 300
