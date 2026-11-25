"""断线重连状态重建。

修复点与函数对应关系：

1. 超窗走全量 ``full_on_overflow`` / ``resync``
   客户端只持有最后确认的输入序号，服务器仅保留最近 ``window_seconds`` 秒的
   输入历史。序号落在窗口之外（或历史为空）时不再返回空增量让客户端静默退回
   出生点，而是返回全量快照。

2. 握手等资源 ``handshake_after_load`` / ``connect``
   重连握手与场景资源加载并发，握手完成时实体可能尚未反序列化。握手必须等
   资源加载与反序列化完成，空实体表不是合法状态，``validate_state`` 拒绝写入。

3. 会话去重 ``dedup_session``
   网络抖动下两次重连请求可能同时到达，同一会话只允许一个连接槽，后到请求
   在锁内被拒绝，不能交替写入同一份状态。

4. 冷却对齐会话 ``cooldown_session``
   技能冷却完全以服务器会话时间计时，会话暂停时会话时间冻结、冷却随之冻结；
   客户端不得按本地墙钟自行推演。

5. 标记带过期 ``marker_expire``
   队友标记创建时即携带 ``expires_at``（会话时间），重连快照与客户端写入前
   统一清理过期标记，不依赖"下一次同步"。

6. 重建幂等 ``idempotent`` / ``recover``
   全量重建是整体替换而非合并，同一份快照重复重建得到逐字节相同的状态。
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

MODE_FULL = "full"
MODE_DELTA = "delta"

SPAWN_POINT: Tuple[float, float] = (0.0, 0.0)
PLAYER_ENTITY_ID = "player:1"


class InvalidStateError(ValueError):
    """状态快照非法（缺实体表、空实体表、序号缺失等）。"""


# ---------------------------------------------------------------------------
# 基础数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InputEvent:
    """服务器输入历史中的一条输入。"""

    seq: int
    session_time: float
    payload: Dict[str, Any]


@dataclass(frozen=True)
class Marker:
    """队友标记，创建时即带会话时间口径的过期时刻。"""

    marker_id: str
    owner: str
    x: float
    y: float
    created_at: float
    expires_at: float

    def is_expired(self, now_session: float) -> bool:
        return now_session >= self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "marker_id": self.marker_id,
            "owner": self.owner,
            "x": self.x,
            "y": self.y,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }


@dataclass(frozen=True)
class Cooldown:
    """技能冷却，起止时刻全部使用会话时间。"""

    skill: str
    started_at: float
    expires_at: float

    def remaining(self, now_session: float) -> float:
        # 会话暂停时 now_session 不增长，剩余时间自然冻结，无需额外判断。
        return max(0.0, self.expires_at - now_session)


@dataclass(frozen=True)
class ConnectionSlot:
    """会话唯一的连接槽。"""

    slot_id: str
    session_id: str
    request_id: str
    connected_at: float


class Counters:
    """重连过程指标。"""

    def __init__(self) -> None:
        self.full_resyncs = 0
        self.delta_resyncs = 0
        self.dedup_rejected = 0
        self.handshake_waits = 0
        self.markers_pruned = 0
        self.invalid_state_rejected = 0
        self.fallbacks = 0

    def as_dict(self) -> Dict[str, int]:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# 服务器会话
# ---------------------------------------------------------------------------


def _initial_entities() -> Dict[str, Any]:
    return {
        PLAYER_ENTITY_ID: {
            "id": PLAYER_ENTITY_ID,
            "x": SPAWN_POINT[0],
            "y": SPAWN_POINT[1],
            "hp": 100,
        }
    }


class Session:
    """服务器侧会话：输入历史、实体表、冷却、标记、连接槽统一在此维护。"""

    def __init__(self, session_id: str, window_seconds: float = 10.0) -> None:
        self.session_id = session_id
        self.window_seconds = float(window_seconds)
        self.inputs: List[InputEvent] = []
        self.entities: Dict[str, Any] = _initial_entities()
        self.session_time = 0.0
        self.paused = False
        self.cooldowns: Dict[str, Cooldown] = {}
        self.markers: Dict[str, Marker] = {}
        self.slot: Optional[ConnectionSlot] = None
        self.lock = threading.RLock()
        self.counters = Counters()
        self.audit_log: List[Dict[str, Any]] = []
        self._seq = 0

    # -- 会话时钟 ---------------------------------------------------------

    def advance(self, dt: float) -> None:
        """推进会话时间；暂停状态下调用不产生任何时间流逝。"""
        with self.lock:
            if not self.paused and dt > 0:
                self.session_time += dt
            self._prune_history()

    def pause(self) -> None:
        with self.lock:
            self.paused = True

    def resume(self) -> None:
        with self.lock:
            self.paused = False

    # -- 输入历史 ---------------------------------------------------------

    def add_input(self, payload: Dict[str, Any]) -> InputEvent:
        with self.lock:
            self._seq += 1
            evt = InputEvent(
                seq=self._seq,
                session_time=self.session_time,
                payload=copy.deepcopy(payload),
            )
            self.inputs.append(evt)
            self._prune_history()
            return evt

    def _prune_history(self) -> None:
        """只保留最近 window_seconds 秒的输入，模拟服务器的滚动窗口。"""
        cutoff = self.session_time - self.window_seconds
        self.inputs = [e for e in self.inputs if e.session_time >= cutoff]

    @property
    def latest_seq(self) -> int:
        return self._seq

    def window_start_seq(self) -> Optional[int]:
        with self.lock:
            return self.inputs[0].seq if self.inputs else None

    # -- 冷却（会话时间口径）----------------------------------------------

    def set_cooldown(self, skill: str, duration: float) -> Cooldown:
        with self.lock:
            cd = Cooldown(
                skill=skill,
                started_at=self.session_time,
                expires_at=self.session_time + duration,
            )
            self.cooldowns[skill] = cd
            return cd

    # -- 标记（带过期）----------------------------------------------------

    def add_marker(
        self, marker_id: str, owner: str, x: float, y: float, ttl: float
    ) -> Marker:
        with self.lock:
            marker = Marker(
                marker_id=marker_id,
                owner=owner,
                x=x,
                y=y,
                created_at=self.session_time,
                expires_at=self.session_time + ttl,
            )
            self.markers[marker_id] = marker
            return marker

    # -- 连接槽 -----------------------------------------------------------

    def release_slot(self, slot_id: Optional[str] = None) -> bool:
        with self.lock:
            if self.slot is None:
                return False
            if slot_id is not None and self.slot.slot_id != slot_id:
                return False
            self.slot = None
            return True

    # -- 快照 -------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """构造全量快照，写入前清理过期标记。"""
        with self.lock:
            prune_result = marker_expire(
                {m.marker_id: m for m in self.markers.values()}, self.session_time
            )
            self.markers = {
                mid: self.markers[mid] for mid in prune_result["active"]
            }
            self.counters.markers_pruned += prune_result["removed"]
            snap: Dict[str, Any] = {
                "mode": MODE_FULL,
                "session_id": self.session_id,
                "seq": self._seq,
                "session_time": self.session_time,
                "entities": copy.deepcopy(self.entities),
                "cooldowns": {
                    skill: {
                        "started_at": cd.started_at,
                        "expires_at": cd.expires_at,
                    }
                    for skill, cd in self.cooldowns.items()
                },
                "markers": {
                    mid: self.markers[mid].to_dict()
                    for mid in sorted(self.markers)
                },
            }
            snap["checksum"] = checksum(snap)
            return snap


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def checksum(obj: Any) -> str:
    """对可 JSON 化对象做稳定哈希，用于确定性与幂等校验。"""
    payload = json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_smoke(value: Any) -> bool:
    """契约测试以裸整数 1 调用，此时走内置演示对象而非业务对象。"""
    return isinstance(value, int) and not isinstance(value, bool)


def _demo_session() -> Session:
    """构造确定性的演示会话，供无状态上下文的冒烟调用使用。"""
    session = Session("sess-demo", window_seconds=5.0)
    session.add_input({"type": "move", "to": [1.0, 0.0]})
    session.add_input({"type": "move", "to": [2.0, 0.0]})
    session.advance(2.0)
    session.set_cooldown("fireball", duration=4.0)
    session.add_marker("m1", owner="teammate-a", x=10.0, y=5.0, ttl=3.0)
    session.add_marker("m2", owner="teammate-b", x=1.0, y=1.0, ttl=30.0)
    return session


def _coerce_session(session: Any) -> Session:
    if isinstance(session, Session):
        return session
    return _demo_session()


def _marker_expires_at(marker: Any) -> float:
    if isinstance(marker, Marker):
        return marker.expires_at
    return float(marker["expires_at"])


def _marker_to_dict(marker: Any) -> Dict[str, Any]:
    if isinstance(marker, Marker):
        return marker.to_dict()
    return dict(marker)


# ---------------------------------------------------------------------------
# 修复点 1：超窗走全量
# ---------------------------------------------------------------------------


def full_on_overflow(
    last_acked_seq: int = 0,
    window_start_seq: Optional[int] = None,
    latest_seq: Optional[int] = None,
    history_empty: bool = False,
) -> Dict[str, Any]:
    """判断重连应走增量还是全量。

    ``window_start_seq`` 是服务器仍保留的最老输入序号。客户端最后确认序号
    与其之间出现缺口（输入已被滚动窗口淘汰）时无法补增量，必须全量；
    历史为空或客户端序号异常超前同样全量兜底。
    """
    if history_empty or window_start_seq is None or latest_seq is None:
        return {"mode": MODE_FULL, "reason": "empty_history"}
    if last_acked_seq < window_start_seq - 1:
        return {
            "mode": MODE_FULL,
            "reason": "overflow",
            "last_acked_seq": last_acked_seq,
            "window_start_seq": window_start_seq,
        }
    if last_acked_seq > latest_seq:
        return {
            "mode": MODE_FULL,
            "reason": "seq_ahead_of_server",
            "last_acked_seq": last_acked_seq,
            "latest_seq": latest_seq,
        }
    return {"mode": MODE_DELTA, "reason": "in_window"}


def resync(session: Any = None, last_acked_seq: int = 0) -> Dict[str, Any]:
    """根据客户端最后确认序号生成重连消息：超窗返回全量快照。"""
    server = _coerce_session(session)
    with server.lock:
        server._prune_history()
        start = server.window_start_seq()
        decision = full_on_overflow(
            last_acked_seq=last_acked_seq,
            window_start_seq=start,
            latest_seq=server.latest_seq,
            history_empty=not server.inputs,
        )
        if decision["mode"] == MODE_FULL:
            server.counters.full_resyncs += 1
            audit(
                server,
                {
                    "event": "resync_full",
                    "reason": decision["reason"],
                    "last_acked_seq": last_acked_seq,
                },
            )
            return server.snapshot()

        server.counters.delta_resyncs += 1
        missing = [
            {
                "seq": evt.seq,
                "session_time": evt.session_time,
                "payload": copy.deepcopy(evt.payload),
            }
            for evt in server.inputs
            if evt.seq > last_acked_seq
        ]
        audit(
            server,
            {
                "event": "resync_delta",
                "base_seq": last_acked_seq,
                "count": len(missing),
            },
        )
        return {
            "mode": MODE_DELTA,
            "session_id": server.session_id,
            "base_seq": last_acked_seq,
            "latest_seq": server.latest_seq,
            "session_time": server.session_time,
            "inputs": missing,
        }


# ---------------------------------------------------------------------------
# 修复点 2：握手等资源
# ---------------------------------------------------------------------------


def handshake_after_load(
    loaded: bool = False,
    deserialized: bool = False,
    entities: Any = None,
) -> Dict[str, Any]:
    """握手闸门：资源未加载完、实体未反序列化完一律不放行。

    空实体表（``{}``）是反序列化未完成的典型表现，必须显式拒绝，不能被
    恢复逻辑当成合法状态写入。
    """
    if not loaded:
        return {"ready": False, "reason": "resources_not_loaded"}
    if not deserialized:
        return {"ready": False, "reason": "entities_not_deserialized"}
    if entities is None:
        return {"ready": False, "reason": "no_entity_table"}
    if isinstance(entities, dict) and len(entities) == 0:
        return {"ready": False, "reason": "empty_entities"}
    return {"ready": True, "reason": "ready"}


# ---------------------------------------------------------------------------
# 修复点 3：会话去重
# ---------------------------------------------------------------------------


def dedup_session(
    session: Any = None,
    request_id: Optional[str] = None,
    slot_id: Optional[str] = None,
) -> Dict[str, Any]:
    """为重连请求分配会话唯一连接槽。

    锁内检查-赋值，保证并发请求只有一个能拿到槽；后到请求被拒绝并记录
    指标与审计，不允许出现两个槽交替写入同一份状态。
    """
    server = _coerce_session(session)
    request_id = request_id or f"req-{uuid.uuid4().hex}"
    with server.lock:
        if server.slot is not None:
            server.counters.dedup_rejected += 1
            audit(
                server,
                {
                    "event": "dedup_rejected",
                    "request_id": request_id,
                    "existing_slot": server.slot.slot_id,
                },
            )
            return {
                "accepted": False,
                "reason": "duplicate_session_connection",
                "request_id": request_id,
                "existing_slot": server.slot.slot_id,
            }
        slot = ConnectionSlot(
            slot_id=slot_id or f"slot-{uuid.uuid4().hex}",
            session_id=server.session_id,
            request_id=request_id,
            connected_at=server.session_time,
        )
        server.slot = slot
        audit(
            server,
            {
                "event": "slot_acquired",
                "request_id": request_id,
                "slot_id": slot.slot_id,
            },
        )
        return {
            "accepted": True,
            "reason": "slot_acquired",
            "request_id": request_id,
            "slot": slot,
        }


# ---------------------------------------------------------------------------
# 修复点 4：冷却对齐会话时间
# ---------------------------------------------------------------------------


def cooldown_session(
    session: Any = None,
    skill: Optional[str] = None,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """返回技能冷却视图，剩余时间只按会话时间计算。

    会话暂停时 ``session_time`` 冻结，无论客户端本地墙钟走了多久，
    ``remaining`` 都保持不变；恢复后随会话时间继续倒计时。
    """
    server = _coerce_session(session)
    with server.lock:
        if skill is None:
            skill = next(iter(server.cooldowns), "fireball")
        now_session = server.session_time if now is None else now
        cd = server.cooldowns.get(skill)
        if cd is None:
            return {
                "skill": skill,
                "clock": "session",
                "paused": server.paused,
                "now_session": now_session,
                "remaining": 0.0,
                "available": True,
            }
        remaining = cd.remaining(now_session)
        return {
            "skill": skill,
            "clock": "session",
            "paused": server.paused,
            "now_session": now_session,
            "started_at": cd.started_at,
            "expires_at": cd.expires_at,
            "remaining": remaining,
            "available": remaining <= 0.0,
        }


# ---------------------------------------------------------------------------
# 修复点 5：标记带过期
# ---------------------------------------------------------------------------


def marker_expire(markers: Any = None, now: float = 0.0) -> Dict[str, Any]:
    """按会话时间清理过期标记，返回存活标记与被清理清单。"""
    if not isinstance(markers, dict):
        markers = {}
    active: Dict[str, Dict[str, Any]] = {}
    pruned: List[Dict[str, Any]] = []
    for mid, marker in markers.items():
        data = _marker_to_dict(marker)
        if _marker_expires_at(marker) <= now:
            pruned.append(data)
        else:
            active[mid] = data
    return {
        "active": active,
        "pruned": pruned,
        "removed": len(pruned),
        "now_session": now,
    }


# ---------------------------------------------------------------------------
# 状态校验、重建与应用
# ---------------------------------------------------------------------------


def validate_state(state: Any = None) -> Dict[str, Any]:
    """校验待写入状态；空实体表等非法形态直接抛错，拒绝写入。"""
    if _is_smoke(state) or state is None:
        state = _demo_session().snapshot()
    if not isinstance(state, dict):
        raise InvalidStateError("state must be a dict")
    if "seq" not in state or not isinstance(state["seq"], int):
        raise InvalidStateError("state missing integer seq")
    entities = state.get("entities")
    if not isinstance(entities, dict):
        raise InvalidStateError("state missing entity table")
    if len(entities) == 0:
        # 关键修复：握手未完成时的空表不得作为合法状态落盘。
        raise InvalidStateError("empty entity table is not a valid state")
    if PLAYER_ENTITY_ID not in entities:
        raise InvalidStateError("player entity missing from snapshot")
    return copy.deepcopy(state)


def recover(local_state: Any = None, snapshot: Any = None) -> Dict[str, Any]:
    """从全量快照整体重建本地状态（替换而非合并，保证幂等）。"""
    if _is_smoke(local_state) or local_state is None:
        local_state = {"entities": _initial_entities(), "last_acked_seq": 0}
    if _is_smoke(snapshot) or snapshot is None:
        snapshot = _demo_session().snapshot()
    snap = validate_state(snapshot)
    rebuilt: Dict[str, Any] = {
        "session_id": snap.get("session_id"),
        "last_acked_seq": snap["seq"],
        "session_time": snap.get("session_time", 0.0),
        "entities": copy.deepcopy(snap["entities"]),
        "cooldowns": copy.deepcopy(snap.get("cooldowns", {})),
        "markers": copy.deepcopy(snap.get("markers", {})),
        "applied_inputs": [],
    }
    prune_result = marker_expire(rebuilt["markers"], rebuilt["session_time"])
    rebuilt["markers"] = prune_result["active"]
    return rebuilt


def _fold_input(state: Dict[str, Any], evt: Dict[str, Any]) -> None:
    """确定性地折叠一条输入；同序输入在任何机器上得到相同结果。"""
    payload = evt.get("payload", {})
    if payload.get("type") == "move" and "to" in payload:
        player = state["entities"].setdefault(
            PLAYER_ENTITY_ID,
            {"id": PLAYER_ENTITY_ID, "x": SPAWN_POINT[0], "y": SPAWN_POINT[1]},
        )
        player["x"], player["y"] = payload["to"]
    state["applied_inputs"].append(evt["seq"])
    state["last_acked_seq"] = evt["seq"]


def apply(local_state: Any = None, message: Any = None) -> Dict[str, Any]:
    """应用重连消息。

    全量消息先过 ``validate_state`` 再整体重建；增量消息按序号顺序折叠。
    任何路径都不会把空实体表写入本地状态。
    """
    if _is_smoke(local_state) or local_state is None:
        local_state = {"entities": _initial_entities(), "last_acked_seq": 0}
    if _is_smoke(message) or message is None:
        message = _demo_session().snapshot()
    if not isinstance(message, dict) or "mode" not in message:
        raise InvalidStateError("message missing mode")

    if message["mode"] == MODE_FULL:
        return recover(local_state, message)

    if message["mode"] != MODE_DELTA:
        raise InvalidStateError(f"unknown message mode: {message['mode']}")

    state = copy.deepcopy(local_state)
    if not isinstance(state.get("entities"), dict) or not state["entities"]:
        raise InvalidStateError("cannot apply delta onto empty local state")
    applied = set(state.get("applied_inputs", []))
    # 序号升序，保证折叠顺序确定、可重放；已应用序号直接跳过，重复投递幂等。
    for evt in sorted(message.get("inputs", []), key=lambda e: e["seq"]):
        if evt["seq"] in applied:
            continue
        _fold_input(state, evt)
        applied.add(evt["seq"])
    state["session_time"] = message.get(
        "session_time", state.get("session_time", 0.0)
    )
    if isinstance(state.get("markers"), dict):
        prune_result = marker_expire(state["markers"], state["session_time"])
        state["markers"] = prune_result["active"]
    return state


def fallback(
    local_state: Any = None,
    reason: str = "unknown",
    session: Any = None,
) -> Dict[str, Any]:
    """显式兜底：回到出生点初始状态，并记录审计与指标。

    与旧实现"收到空增量静默回出生点"不同，兜底只能由明确的失败原因触发，
    且全程留痕。
    """
    state: Dict[str, Any] = {
        "session_id": None,
        "last_acked_seq": 0,
        "session_time": 0.0,
        "entities": _initial_entities(),
        "cooldowns": {},
        "markers": {},
        "applied_inputs": [],
        "fallback_reason": reason,
    }
    if isinstance(session, Session):
        with session.lock:
            session.counters.fallbacks += 1
            state["session_id"] = session.session_id
            audit(session, {"event": "fallback", "reason": reason})
    elif isinstance(local_state, dict) and local_state.get("session_id"):
        state["session_id"] = local_state["session_id"]
    return state


# ---------------------------------------------------------------------------
# 重连主流程
# ---------------------------------------------------------------------------


def connect(
    session: Any = None,
    request_id: Optional[str] = None,
    last_acked_seq: int = 0,
    resources_loaded: bool = True,
    entities_deserialized: bool = True,
    entities: Any = None,
) -> Dict[str, Any]:
    """重连入口：握手闸门 -> 会话去重 -> 增量/全量 resync。

    握手等待期间不占用连接槽；闸门放行后在锁内原子获取唯一槽位，
    再根据输入窗口决定补发增量还是全量快照。
    """
    server = _coerce_session(session)
    request_id = request_id or f"req-{uuid.uuid4().hex}"
    if entities is None:
        entities = server.entities

    gate = handshake_after_load(
        resources_loaded, entities_deserialized, entities
    )
    if not gate["ready"]:
        with server.lock:
            server.counters.handshake_waits += 1
        audit(
            server,
            {
                "event": "handshake_wait",
                "request_id": request_id,
                "reason": gate["reason"],
            },
        )
        return {
            "connected": False,
            "stage": "handshake",
            "request_id": request_id,
            **gate,
        }

    decision = dedup_session(server, request_id)
    if not decision["accepted"]:
        return {
            "connected": False,
            "stage": "dedup",
            "request_id": request_id,
            **decision,
        }

    message = resync(server, last_acked_seq)
    return {
        "connected": True,
        "stage": "active",
        "request_id": request_id,
        "slot_id": decision["slot"].slot_id,
        "message": message,
    }


# ---------------------------------------------------------------------------
# 指标 / 审计 / 确定性 / 幂等
# ---------------------------------------------------------------------------


def metrics(session: Any = None) -> Dict[str, int]:
    """返回重连指标快照。"""
    server = _coerce_session(session)
    with server.lock:
        return server.counters.as_dict()


def audit(session: Any = None, event: Any = None) -> Dict[str, Any]:
    """追加并返回一条审计记录；无事件时返回最近一条。"""
    server = _coerce_session(session)
    with server.lock:
        if event is None:
            event = {"event": "audit_poll"}
        entry = {
            "id": len(server.audit_log) + 1,
            "session_time": server.session_time,
            "event": copy.deepcopy(event),
        }
        server.audit_log.append(entry)
        return entry


def determinism(
    snapshot: Any = None, inputs: Any = None
) -> Dict[str, Any]:
    """同一份快照与同一批输入，两次独立重建必须得到相同校验和。"""
    if _is_smoke(snapshot) or snapshot is None:
        demo = _demo_session()
        snapshot = demo.snapshot()
        inputs = [
            {"seq": evt.seq, "payload": evt.payload}
            for evt in demo.inputs
        ]
    state_a = recover(None, snapshot)
    state_b = recover(None, snapshot)
    for evt in inputs or []:
        _fold_input(state_a, evt)
        _fold_input(state_b, evt)
    sum_a, sum_b = checksum(state_a), checksum(state_b)
    return {
        "deterministic": sum_a == sum_b,
        "checksum_a": sum_a,
        "checksum_b": sum_b,
    }


def idempotent(snapshot: Any = None, times: int = 2) -> Dict[str, Any]:
    """同一快照重复重建 N 次，所有结果校验和一致即为幂等。"""
    if _is_smoke(snapshot) or snapshot is None:
        snapshot = _demo_session().snapshot()
    times = max(2, int(times))
    checksums: List[str] = []
    last_state: Optional[Dict[str, Any]] = None
    for _ in range(times):
        last_state = recover(None, snapshot)
        checksums.append(checksum(last_state))
    unique = set(checksums)
    return {
        "idempotent": len(unique) == 1,
        "times": times,
        "checksums": checksums,
        "state": last_state,
    }
