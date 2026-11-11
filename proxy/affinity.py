"""Route a pool of single-slot llama.cpp replicas by recent prompt history."""

import asyncio
from dataclasses import dataclass, field
from hashlib import sha256
import json
import time

import httpx


def fingerprint(body):
    return tuple(
        (message.get("role"), sha256(json.dumps(
            message, sort_keys=True, separators=(",", ":")
        ).encode()).digest())
        for message in body.get("messages") or []
    )


@dataclass
class Replica:
    base_url: str
    model: str
    busy: bool = False
    last_used: float = 0
    session: str | None = None
    history: tuple = field(default_factory=tuple)

    def score(self, session, history):
        shared = 0
        has_turn = False
        for old, new in zip(self.history, history):
            if old != new:
                break
            shared += 1
            has_turn |= new[0] not in ("system", "developer")
        # Shared system instructions alone say little about conversation identity.
        return has_turn, shared if has_turn else 0, self.session == session


class Pool:
    def __init__(self, specs):
        self.replicas = [Replica(spec["base_url"], spec["model"]) for spec in specs]
        self.changed = asyncio.Condition()

    async def acquire(self, client, session, history):
        async def ready(replica):
            try:
                result = await client.get(replica.base_url.rstrip("/") + "/health", timeout=1)
                return result.status_code == 200
            except httpx.RequestError:
                return False

        while True:
            health = await asyncio.gather(*(ready(replica) for replica in self.replicas))
            async with self.changed:
                available = []
                for replica, ok in zip(self.replicas, health):
                    if ok:
                        available.append(replica)
                    elif not replica.busy:
                        replica.history = ()
                        replica.session = None
                if not available:
                    return None
                best = max(replica.score(session, history) for replica in available)
                candidates = [replica for replica in available if not replica.busy
                              and replica.score(session, history) == best]
                if candidates:
                    replica = min(candidates, key=lambda item: item.last_used)
                    replica.busy = True
                    replica.session = session
                    replica.history = history
                    return replica
                try:
                    await asyncio.wait_for(self.changed.wait(), timeout=1)
                except TimeoutError:
                    pass

    async def release(self, replica, success):
        async with self.changed:
            if not success:
                replica.history = ()
                replica.session = None
            replica.last_used = time.monotonic()
            replica.busy = False
            self.changed.notify_all()


class ReleaseReplica:
    """Hold a replica until the ASGI response, including streaming, finishes."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        status = 0
        complete = False

        async def track(message):
            nonlocal status, complete
            await send(message)
            if message["type"] == "http.response.start":
                status = message["status"]
            if message["type"] == "http.response.body" and not message.get("more_body"):
                complete = True

        try:
            await self.app(scope, receive, track)
        finally:
            lease = scope.pop("minion_lease", None)
            if lease:
                pool, replica = lease
                await asyncio.shield(pool.release(
                    replica, complete and 200 <= status < 300 and scope.get("minion_forwarded")
                ))
