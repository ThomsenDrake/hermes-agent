"""An unanswered gateway notice must not swallow the next human message.

A notice is stored as role=user, with a footer that tells the model it was not written
by a person (gateway.run_notifications). When no assistant reply follows it, alternation
repair used to concatenate the next human turn onto that row. The person's request then
sat under the non-human footer and the model refused it. Resending worked, because that
second copy no longer sat directly against the unanswered notice.

Either order is the same bug: a notice welded onto an open human turn stamps the footer
on the person's text.
"""

import pytest

from agent.agent_runtime_helpers import repair_message_sequence
from agent.agent_runtime_helpers_user_merge import (
    NOTIFICATION_TURN_BOUNDARY,
    _merge_consecutive_users,
    _notification_boundary,
)
from agent.message_metadata import NOTIFICATION_BOUNDARY_FLAG, without_persistence_fields
from agent.session_persistence import _is_ephemeral_scaffolding
from gateway.response_filters import (
    INTERNAL_NOTIFICATION_DISPLAY_KIND,
    MACHINERY_DISPLAY_KINDS,
)
from gateway.run_notifications import INTERNAL_NOTIFICATION_FOOTER
from hermes_state import SessionDB

HUMAN = "rebuild the release from the current tree"
NOTICE = "Background process finished: make test\n\n" + INTERNAL_NOTIFICATION_FOOTER


def _notice_body(kind: str) -> str:
    """The reported notice carries the non-human footer; other kinds are identified by display kind."""
    if kind == INTERNAL_NOTIFICATION_DISPLAY_KIND:
        return NOTICE
    return f"{kind} finished"


def _notice(kind: str = INTERNAL_NOTIFICATION_DISPLAY_KIND, content: str | None = None) -> dict:
    return {"role": "user", "content": NOTICE if content is None else content, "display_kind": kind}


def _human(content: str = HUMAN) -> dict:
    return {"role": "user", "content": content}


def _user_contents(messages: list[dict]) -> list[str]:
    return [m["content"] for m in messages if m.get("role") == "user"]


def _assert_separated(messages: list[dict], notice: str, human: str) -> None:
    contents = _user_contents(messages)
    assert contents.count(notice) == 1
    assert contents.count(human) == 1
    assert all(not (notice in c and human in c) for c in contents)
    roles = [m["role"] for m in messages]
    assert not any(a == "user" and b == "user" for a, b in zip(roles, roles[1:]))
    notice_at = next(i for i, m in enumerate(messages) if m.get("content") == notice)
    human_at = next(i for i, m in enumerate(messages) if m.get("content") == human)
    lo, hi = sorted((notice_at, human_at))
    between = messages[lo + 1:hi]
    assert [m.get("content") for m in between] == [NOTIFICATION_TURN_BOUNDARY]
    seam = between[0]
    assert seam["role"] == "assistant"
    assert seam["display_kind"] == "hidden"
    assert _is_ephemeral_scaffolding(seam)


@pytest.mark.parametrize("kind", sorted(MACHINERY_DISPLAY_KINDS))
@pytest.mark.parametrize("notice_first", [True, False])
def test_human_turn_is_not_merged_into_a_gateway_notice(kind, notice_first):
    """The reported case, and the same weld in the other order, for every notice kind."""
    notice = _notice(kind, _notice_body(kind))
    human = _human()
    before_notice, before_human = notice["content"], human["content"]
    pair = [notice, human] if notice_first else [human, notice]

    merged, repairs = _merge_consecutive_users(pair)

    assert repairs == 1
    assert notice["content"] == before_notice
    assert human["content"] == before_human
    _assert_separated(merged, before_notice, before_human)
    if INTERNAL_NOTIFICATION_FOOTER in before_notice:
        human_row = next(m for m in merged if m.get("content") == before_human)
        assert INTERNAL_NOTIFICATION_FOOTER not in human_row["content"]
    again, more = _merge_consecutive_users(merged)
    assert more == 0
    assert [m["content"] for m in again] == [m["content"] for m in merged]


def test_footer_alone_keeps_a_human_turn_out_from_under_it():
    """A notice whose display kind was already dropped is still the footer's row."""
    messages = [{"role": "user", "content": NOTICE}, _human()]
    repairs = repair_message_sequence(None, messages)

    assert repairs == 1
    _assert_separated(messages, NOTICE, HUMAN)
    assert INTERNAL_NOTIFICATION_FOOTER not in messages[-1]["content"]


def test_unanswered_notice_then_human_message_repairs_on_restore(tmp_path):
    """Durable shape of the bug: a notice row, no assistant reply, then the next human message."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s", "system")
        db.append_message(session_id="s", role="user", content="earlier question")
        db.append_message(session_id="s", role="assistant", content="earlier answer")
        db.append_message(
            session_id="s", role="user", content=NOTICE,
            display_kind=INTERNAL_NOTIFICATION_DISPLAY_KIND,
        )
        db.append_message(session_id="s", role="user", content=HUMAN)

        messages = db.get_messages_as_conversation("s", repair_alternation=True)
    finally:
        db.close()

    assert repair_message_sequence(None, messages) == 0
    _assert_separated(messages, NOTICE, HUMAN)
    assert messages[0]["content"] == "earlier question"
    assert messages[1]["content"] == "earlier answer"


def test_human_then_notice_restores_without_stamping_the_footer_on_the_human(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("s", "system")
        db.append_message(session_id="s", role="user", content=HUMAN)
        db.append_message(
            session_id="s", role="user", content=NOTICE,
            display_kind=INTERNAL_NOTIFICATION_DISPLAY_KIND,
        )
        messages = db.get_messages_as_conversation("s", repair_alternation=True)
    finally:
        db.close()

    assert repair_message_sequence(None, messages) == 0
    _assert_separated(messages, NOTICE, HUMAN)
    human_row = next(m for m in messages if m.get("content") == HUMAN)
    assert INTERNAL_NOTIFICATION_FOOTER not in human_row["content"]


def test_two_plain_human_turns_still_merge():
    merged, repairs = _merge_consecutive_users([_human("first"), _human("second")])
    assert repairs == 1
    assert len(merged) == 1
    assert merged[0]["role"] == "user"
    assert merged[0]["content"] == "first\n\nsecond"


def test_two_notices_still_fold_so_the_provider_does_not_see_user_user():
    first = _notice(content="one finished\n\n" + INTERNAL_NOTIFICATION_FOOTER)
    second = _notice(content="two finished\n\n" + INTERNAL_NOTIFICATION_FOOTER)
    merged, repairs = _merge_consecutive_users([first, second])
    assert repairs == 1
    assert [m["role"] for m in merged] == ["user"]
    assert "one finished" in merged[0]["content"]
    assert "two finished" in merged[0]["content"]
    assert merged[0]["display_kind"] == INTERNAL_NOTIFICATION_DISPLAY_KIND


def test_notice_boundary_is_not_provider_payload_and_is_not_flushed():
    boundary = _notification_boundary()
    wire = without_persistence_fields(boundary)
    assert wire == {"role": "assistant", "content": NOTIFICATION_TURN_BOUNDARY}
    assert NOTIFICATION_BOUNDARY_FLAG not in wire
    assert _is_ephemeral_scaffolding(boundary)
