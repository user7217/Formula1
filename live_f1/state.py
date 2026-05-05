import asyncio
import time
from typing import Any, Set


class StateManager:
    """Single in-memory source of truth. Holds latest per-topic state and
    broadcasts changes to all subscribed asyncio queues."""

    MAX_RACE_CONTROL = 200

    def __init__(self):
        self._state: dict[str, Any] = {
            "session": {"status": "idle"},
            "drivers": {},
            "timing": {},
            "telemetry": {},
            "positions": {},
            "weather": {},
            "race_control": [],
            "track_status": "Green",
        }
        self._subscribers: Set[asyncio.Queue] = set()

    def get(self, key: str | None = None):
        if key is None:
            return self._state
        return self._state.get(key)

    def snapshot(self) -> dict:
        return self._state

    async def set_topic(self, topic: str, data: Any):
        self._state[topic] = data
        await self._broadcast(topic, data)

    async def append_race_control(self, msg: dict):
        rc = self._state["race_control"]
        rc.append(msg)
        if len(rc) > self.MAX_RACE_CONTROL:
            del rc[: len(rc) - self.MAX_RACE_CONTROL]
        await self._broadcast("race_control_msg", msg)

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        self._subscribers.discard(q)

    async def _broadcast(self, topic: str, data: Any):
        msg = {"topic": topic, "data": data, "ts": time.time()}
        dead = []
        for q in self._subscribers:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self._subscribers.discard(q)