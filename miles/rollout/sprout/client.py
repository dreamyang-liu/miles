"""HTTP client for Sprout's RL driver."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

import httpx

from miles.rollout.sprout.protocol import TERMINAL_GROUP_STATUSES, Acknowledgement, RolloutRequest, RolloutResult


class SproutDriverError(RuntimeError):
    """The driver answered with an error status; the body says why."""

    def __init__(self, method: str, path: str, response: httpx.Response) -> None:
        self.status_code = response.status_code
        super().__init__(f"Sprout driver {method} {path} -> HTTP {response.status_code}: {response.text[:500]}")


class SproutRolloutClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout: float | None = 30.0,
        token: str | None = None,
        headers: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        merged = dict(headers or {})
        if token:
            merged["Authorization"] = f"Bearer {token}"
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=httpx.Timeout(timeout), headers=merged
        )

    async def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        response = await self._client.request(method, path, **kwargs)
        if response.status_code >= 400:
            raise SproutDriverError(method, path, response)
        return response.json()

    async def submit(self, request: RolloutRequest) -> Acknowledgement:
        body = await self._call("POST", "/rollout-groups", json=request.model_dump(mode="json"))
        return Acknowledgement.model_validate(body)

    async def get_result(self, rollout_job_id: str) -> RolloutResult:
        return RolloutResult.model_validate(await self._call("GET", f"/rollout-groups/{rollout_job_id}"))

    async def wait_for_result(self, rollout_job_id: str, *, poll_interval_seconds: float) -> RolloutResult:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be greater than zero")
        while True:
            result = await self.get_result(rollout_job_id)
            if result.rollout_job_id != rollout_job_id:
                raise ValueError(f"driver returned {result.rollout_job_id!r} while polling {rollout_job_id!r}")
            if result.status in TERMINAL_GROUP_STATUSES:
                return result
            await asyncio.sleep(poll_interval_seconds)

    async def delete(self, rollout_job_id: str) -> Acknowledgement:
        """Cancel unfinished work, or acknowledge a terminal group as consumed."""
        body = await self._call("DELETE", f"/rollout-groups/{rollout_job_id}")
        acknowledgement = Acknowledgement.model_validate(body)
        if acknowledgement.rollout_job_id != rollout_job_id or acknowledgement.status not in TERMINAL_GROUP_STATUSES:
            raise ValueError(f"driver did not confirm the release of {rollout_job_id!r}: {body}")
        return acknowledgement

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> SproutRolloutClient:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.close()
