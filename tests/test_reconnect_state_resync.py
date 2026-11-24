import pytest
import gameplay.reconnect_state_resync as mod

CASES = [
    ("full_on_overflow", (1,)),
    ("handshake_after_load", (1,)),
    ("dedup_session", (1,)),
    ("cooldown_session", (1,)),
    ("marker_expire", (1,)),
    ("connect", (1,)),
    ("resync", (1,)),
    ("apply", (1,)),
    ("recover", (1,)),
    ("validate_state", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
    ("idempotent", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
