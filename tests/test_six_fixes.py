"""六个修复点的行为测试，覆盖题目描述的每一个故障场景。"""

import threading

import pytest

from gameplay.reconnect_state_resync import (
    InvalidStateError,
    MODE_DELTA,
    MODE_FULL,
    Session,
    apply,
    audit,
    cooldown_session,
    connect,
    dedup_session,
    determinism,
    fallback,
    full_on_overflow,
    handshake_after_load,
    idempotent,
    marker_expire,
    metrics,
    recover,
    resync,
    validate_state,
)


# ---------------------------------------------------------------------------
# 修复点 1：超窗走全量
# ---------------------------------------------------------------------------


def test_in_window_returns_delta():
    session = Session("s1", window_seconds=10.0)
    for _ in range(3):
        session.add_input({"type": "ping"})
    session.advance(1.0)
    # 最后确认序号在保留窗口内，且后续确有缺失输入 -> 增量
    msg = resync(session, last_acked_seq=1)
    assert msg["mode"] == MODE_DELTA
    assert [e["seq"] for e in msg["inputs"]] == [2, 3]


def test_overflow_returns_full_not_empty_delta():
    session = Session("s1", window_seconds=5.0)
    for _ in range(3):
        session.add_input({"type": "move", "to": [1.0, 0.0]})
    # 让窗口滚动，淘汰早期输入
    session.advance(6.0)
    session.add_input({"type": "move", "to": [2.0, 0.0]})
    start = session.window_start_seq()
    decision = full_on_overflow(
        last_acked_seq=1,
        window_start_seq=start,
        latest_seq=session.latest_seq,
    )
    assert decision == {
        "mode": MODE_FULL,
        "reason": "overflow",
        "last_acked_seq": 1,
        "window_start_seq": start,
    }
    msg = resync(session, last_acked_seq=1)
    # 超窗必须给全量快照，且快照是合法状态而不是空增量
    assert msg["mode"] == MODE_FULL
    assert msg["entities"]
    assert msg["seq"] == session.latest_seq
    assert metrics(session)["full_resyncs"] == 1


def test_empty_history_goes_full_and_player_keeps_position():
    session = Session("s1")
    session.advance(20.0)  # 历史全部过期
    msg = resync(session, last_acked_seq=99)
    assert msg["mode"] == MODE_FULL
    assert msg["entities"]["player:1"] is not None


def test_seq_ahead_of_server_goes_full():
    decision = full_on_overflow(
        last_acked_seq=100, window_start_seq=1, latest_seq=5
    )
    assert decision["mode"] == MODE_FULL
    assert decision["reason"] == "seq_ahead_of_server"


# ---------------------------------------------------------------------------
# 修复点 2：握手等资源
# ---------------------------------------------------------------------------


def test_handshake_blocks_until_load_and_deserialize():
    assert handshake_after_load(False, False, {"e": 1})["ready"] is False
    assert handshake_after_load(True, False, {"e": 1})["ready"] is False
    ready = handshake_after_load(True, True, {"player:1": {}})
    assert ready["ready"] is True


def test_empty_entity_table_rejected_by_handshake_and_validate():
    gate = handshake_after_load(True, True, {})
    assert gate["ready"] is False
    assert gate["reason"] == "empty_entities"
    with pytest.raises(InvalidStateError):
        validate_state({"seq": 1, "entities": {}})
    with pytest.raises(InvalidStateError):
        validate_state({"entities": {"player:1": {}}})  # 缺 seq


def test_connect_waits_for_resources_and_does_not_take_slot():
    session = Session("s2")
    result = connect(
        session,
        request_id="req-wait",
        resources_loaded=False,
        entities_deserialized=False,
    )
    assert result["connected"] is False
    assert result["stage"] == "handshake"
    assert session.slot is None  # 等待期间不占槽
    assert metrics(session)["handshake_waits"] == 1


def test_connect_full_flow_after_load():
    session = Session("s2")
    session.add_input({"type": "ping"})
    result = connect(session, request_id="req-ok", last_acked_seq=0)
    assert result["connected"] is True
    assert result["slot_id"] == session.slot.slot_id
    assert result["message"]["mode"] == MODE_DELTA


# ---------------------------------------------------------------------------
# 修复点 3：会话去重
# ---------------------------------------------------------------------------


def test_duplicate_reconnect_only_one_slot():
    session = Session("s3")
    first = dedup_session(session, request_id="r1")
    second = dedup_session(session, request_id="r2")
    assert first["accepted"] is True
    assert second["accepted"] is False
    assert second["reason"] == "duplicate_session_connection"
    assert session.slot.request_id == "r1"
    assert metrics(session)["dedup_rejected"] == 1


def test_concurrent_reconnects_single_winner():
    session = Session("s3")
    outcomes = []

    def worker(rid):
        outcomes.append(dedup_session(session, request_id=rid)["accepted"])

    threads = [
        threading.Thread(target=worker, args=(f"r{i}",)) for i in range(20)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert outcomes.count(True) == 1
    assert outcomes.count(False) == 19
    assert metrics(session)["dedup_rejected"] == 19


def test_released_slot_allows_reconnect():
    session = Session("s3")
    first = dedup_session(session, request_id="r1")
    assert session.release_slot(first["slot"].slot_id) is True
    second = dedup_session(session, request_id="r2")
    assert second["accepted"] is True


def test_two_connects_same_session_second_rejected():
    session = Session("s3")
    a = connect(session, request_id="a")
    b = connect(session, request_id="b")
    assert a["connected"] is True
    assert b["connected"] is False
    assert b["stage"] == "dedup"


# ---------------------------------------------------------------------------
# 修复点 4：冷却对齐会话时间
# ---------------------------------------------------------------------------


def test_cooldown_freezes_when_session_paused():
    session = Session("s4")
    session.set_cooldown("fireball", duration=10.0)
    view_before = cooldown_session(session, "fireball")
    assert view_before["remaining"] == pytest.approx(10.0)

    session.pause()
    session.advance(8.0)  # 本地墙钟在走，会话时间冻结
    view_paused = cooldown_session(session, "fireball")
    assert view_paused["paused"] is True
    assert view_paused["remaining"] == pytest.approx(10.0)
    assert view_paused["clock"] == "session"

    session.resume()
    session.advance(4.0)
    view_resumed = cooldown_session(session, "fireball")
    assert view_resumed["remaining"] == pytest.approx(6.0)
    assert view_resumed["available"] is False

    session.advance(6.0)
    assert cooldown_session(session, "fireball")["available"] is True


# ---------------------------------------------------------------------------
# 修复点 5：标记带过期
# ---------------------------------------------------------------------------


def test_marker_expire_prunes_by_session_time():
    session = Session("s5")
    session.add_marker("m1", "a", 0, 0, ttl=3.0)
    session.add_marker("m2", "b", 1, 1, ttl=30.0)
    result = marker_expire(
        {m.marker_id: m for m in session.markers.values()}, now=5.0
    )
    assert set(result["active"]) == {"m2"}
    assert result["removed"] == 1
    assert result["pruned"][0]["marker_id"] == "m1"


def test_snapshot_and_recover_drop_expired_markers():
    session = Session("s5")
    session.add_marker("m1", "a", 0, 0, ttl=2.0)
    session.add_marker("m2", "b", 1, 1, ttl=20.0)
    session.advance(3.0)
    snap = session.snapshot()
    assert set(snap["markers"]) == {"m2"}

    rebuilt = recover(None, snap)
    assert set(rebuilt["markers"]) == {"m2"}


def test_expired_marker_not_revived_by_delta_apply():
    local = {
        "entities": {"player:1": {"id": "player:1", "x": 0, "y": 0}},
        "last_acked_seq": 1,
        "session_time": 0.0,
        "markers": {
            "m1": {
                "marker_id": "m1",
                "owner": "a",
                "x": 0,
                "y": 0,
                "created_at": 0.0,
                "expires_at": 2.0,
            }
        },
        "applied_inputs": [],
    }
    msg = {
        "mode": MODE_DELTA,
        "base_seq": 1,
        "latest_seq": 2,
        "session_time": 5.0,
        "inputs": [
            {"seq": 2, "session_time": 5.0, "payload": {"type": "ping"}}
        ],
    }
    out = apply(local, msg)
    assert out["markers"] == {}


# ---------------------------------------------------------------------------
# 修复点 6：重建幂等 + 确定性
# ---------------------------------------------------------------------------


def test_recover_is_full_replacement_not_merge():
    old = {
        "entities": {
            "player:1": {"id": "player:1", "x": 99.0, "y": 99.0},
            "stale:ghost": {"id": "stale:ghost"},
        },
        "last_acked_seq": 50,
        "session_time": 100.0,
        "markers": {"old": {"marker_id": "old", "expires_at": 1.0}},
        "cooldowns": {"old": {"expires_at": 1.0}},
        "applied_inputs": [1, 2, 3],
    }
    session = Session("s6")
    session.add_input({"type": "move", "to": [3.0, 4.0]})
    snap = session.snapshot()
    rebuilt = recover(old, snap)
    assert "stale:ghost" not in rebuilt["entities"]
    assert rebuilt["applied_inputs"] == []  # 替换而非合并
    assert rebuilt["last_acked_seq"] == snap["seq"]
    # 原对象不被修改
    assert "stale:ghost" in old["entities"]


def test_idempotent_rebuild():
    snap = _demo_snapshot()
    result = idempotent(snap, times=5)
    assert result["idempotent"] is True
    assert len(set(result["checksums"])) == 1


def test_determinism_across_independent_rebuilds():
    session = Session("s6")
    inputs = []
    for i in range(4):
        evt = session.add_input(
            {"type": "move", "to": [float(i), float(i * 2)]}
        )
        inputs.append({"seq": evt.seq, "payload": evt.payload})
    snap = session.snapshot()
    result = determinism(snap, inputs)
    assert result["deterministic"] is True


def test_apply_delta_twice_does_not_diverge():
    session = Session("s6")
    session.add_input({"type": "move", "to": [7.0, 8.0]})
    msg = resync(session, last_acked_seq=0)
    base = {"entities": session.entities.copy(), "last_acked_seq": 0,
            "markers": {}, "applied_inputs": [], "session_time": 0.0}
    once = apply(base, msg)
    twice = apply(apply(base, msg), msg)
    # 第二次没有新输入，状态不发生漂移
    assert twice["entities"] == once["entities"]
    assert twice["last_acked_seq"] == once["last_acked_seq"]


def test_fallback_is_explicit_and_audited():
    session = Session("s7")
    out = fallback(None, reason="overflow_unrecoverable", session=session)
    assert out["entities"]["player:1"]["x"] == 0.0
    assert out["fallback_reason"] == "overflow_unrecoverable"
    assert metrics(session)["fallbacks"] == 1
    assert session.audit_log[-1]["event"]["event"] == "fallback"


def test_audit_trail_records_key_decisions():
    session = Session("s8")
    session.add_input({"type": "ping"})
    session.advance(20.0)
    resync(session, last_acked_seq=0)  # 超窗全量
    dedup_session(session, request_id="r1")
    dedup_session(session, request_id="r2")  # 被拒
    events = [e["event"]["event"] for e in session.audit_log]
    assert "resync_full" in events
    assert "slot_acquired" in events
    assert "dedup_rejected" in events


def _demo_snapshot():
    session = Session("snap")
    session.add_input({"type": "move", "to": [1.0, 2.0]})
    session.set_cooldown("ice", 5.0)
    session.add_marker("m", "a", 1, 1, ttl=9.0)
    return session.snapshot()
