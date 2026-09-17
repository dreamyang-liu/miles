"""Fail closed if Ray loads the old rollout source or loses adapter/cap fields."""

import asyncio
from functools import lru_cache
import hashlib
import json
import logging
import os
from pathlib import Path
import time

import miles.rollout.ash.message_rollout as implementation
import httpx
from miles.rollout.ash.message_rollout import AshMessageClient, AshMessageRolloutFn
from miles.rollout.ash.message_protocol import AshMessageRequest


expected = Path(os.environ["MILES_NIGHT_SOURCE_ROOT"]).resolve()
assert expected in Path(implementation.__file__).resolve().parents, (
    f"Wrong rollout source: {implementation.__file__}; expected {expected}"
)


@lru_cache(maxsize=1)
def recovery_manifest():
    path = os.environ.get("MILES_NIGHT_RECOVER_REQUESTS")
    if not path:
        return None
    data = Path(path).read_bytes()
    value = json.loads(data)
    assert value["optimizer_updates"] == 0 and value["speculative_decoding"] is False
    return {
        "by_task": {request["task_id"]: request for request in value["requests"]},
        "ids": {request["rollout_job_id"] for request in value["requests"]},
        "sha256": hashlib.sha256(data).hexdigest(),
    }


class RecordingClient(AshMessageClient):
    async def _retry_transport(self, call, *args):
        deadline = time.monotonic() + 300
        attempt = 0
        while True:
            attempt += 1
            try:
                return await call(*args)
            except (httpx.TransportError, httpx.HTTPStatusError) as error:
                if isinstance(error, httpx.HTTPStatusError) and error.response.status_code not in {
                    408, 429, 500, 502, 503, 504,
                }:
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        f"Ash request remained unavailable for 300 seconds ({type(error).__name__})"
                    ) from error
                logging.getLogger(__name__).warning(
                    "Retrying the same Ash request after %s (attempt %s, %.1fs grace left)",
                    type(error).__name__, attempt, remaining,
                )
                await asyncio.sleep(min(2, remaining))

    async def submit(self, request):
        assert request.model.endswith(":miles_lora"), "Live actor request must select the adapter"
        assert request.branching and request.max_sequence_tokens in {49152, 65536, 81920, 131072}
        assert request.max_samples == 8 and request.minimum_returned_samples == 1, (
            "Variable task batches must request up to eight trajectories without requiring a fixed pair"
        )
        output = Path(os.environ["MILES_NIGHT_RUN_DIR"]) / "ash-requests"
        output.mkdir(exist_ok=True)
        (output / (request.rollout_job_id + ".json")).write_text(
            json.dumps(request.to_wire(), ensure_ascii=False, indent=2) + "\n"
        )
        recovery = recovery_manifest()
        if recovery and request.rollout_job_id in recovery["ids"]:
            # Submission is gated until the host verifies inference is ready
            # and restores only the corresponding interrupted Ash jobs.
            marker = output.parent / "recovery-ready.json"
            for _ in range(1800):
                if marker.exists():
                    ready = json.loads(marker.read_text())
                    assert ready["request_sha256"] == recovery["sha256"]
                    assert ready["status"] == "ready"
                    break
                await asyncio.sleep(1)
            else:
                raise TimeoutError("Verified Ash recovery was not made ready")
        return await self._retry_transport(super().submit, request)

    async def get_result(self, rollout_job_id):
        return await self._retry_transport(super().get_result, rollout_job_id)

    async def delete(self, rollout_job_id):
        return await self._retry_transport(super().delete, rollout_job_id)


class NightAshMessageRolloutFn(AshMessageRolloutFn):
    def __init__(self, input):
        assert input.args.rollout_batch_size == 32
        assert not input.args.sglang_speculative_algorithm
        super().__init__(input, client_factory=RecordingClient)

    def _build_request(self, **kwargs):
        request, slots = super()._build_request(**kwargs)
        recovery = recovery_manifest()
        if not recovery or kwargs["rollout_id"] != 0:
            return request, slots
        old_request = recovery["by_task"].get(request.task_id)
        if old_request is None:
            return request, slots
        recovered = AshMessageRequest.model_validate(old_request)
        expected_body = recovered.to_wire()
        actual_body = request.to_wire()
        for body in (expected_body, actual_body):
            body.pop("rollout_job_id")
            body.pop("sample_slots")
        assert expected_body == actual_body, "Recovery prompt/model/settings differ from the new request"
        by_index = {sample.index: sample for sample in kwargs["group"]}
        assert set(by_index) == {slot.sample_index for slot in recovered.sample_slots}
        return recovered, {slot.sample_slot_id: by_index[slot.sample_index] for slot in recovered.sample_slots}
