"""Consecutive user-turn alternation: fold plain human turns, never a gateway notice.

Split out of ``agent/agent_runtime_helpers.py`` (file-size ratchet). The repair pass
imports :func:`_merge_consecutive_users`. This module does not import
``agent_runtime_helpers`` at load time (the absorbed-row helper is imported inside
the fold).
"""

from __future__ import annotations

from typing import Any

# Structural label, not prose. The row is an assistant turn the model reads as its own,
# and a short sentence in that seat is echoed verbatim (#132949).
NOTIFICATION_TURN_BOUNDARY = "[internal-notification delivered]"


def _notification_boundary() -> dict:
    """Assistant seam between a gateway notice and a human turn. Hidden in transcripts,
    dropped from the wire's bookkeeping keys, and never flushed (see the flag)."""
    from agent.message_metadata import NOTIFICATION_BOUNDARY_FLAG

    return {
        "role": "assistant",
        "content": NOTIFICATION_TURN_BOUNDARY,
        "display_kind": "hidden",
        NOTIFICATION_BOUNDARY_FLAG: True,
    }


def _plain_user_text(content: Any) -> bool:
    """Plain text the pass may rewrite. Sentinel-encoded multimodal content stays put."""
    from hermes_state import SessionDB

    return isinstance(content, str) and not content.startswith(SessionDB._CONTENT_JSON_PREFIX)


def _consecutive_plain_users(prev: Any, msg: Any) -> bool:
    """Both rows are plain-text user turns this pass may touch.

    A summary carrier followed by a new user row is a deliberate durable shape after
    retry/rewind; the carrier is left alone. Multimodal rows are left alone too.
    """
    if not (
        isinstance(prev, dict) and prev.get("role") == "user"
        and isinstance(msg, dict) and msg.get("role") == "user"
    ):
        return False
    from agent.context_compressor import split_user_originated_turn

    if split_user_originated_turn(prev)[0] is not None:
        return False
    return _plain_user_text(prev.get("content", "")) and _plain_user_text(msg.get("content", ""))


def _is_gateway_notification(msg: dict) -> bool:
    """A role=user row the gateway wrote as a machine notice, not a person.

    Machinery display kinds are the durable mark. The footer is the mark the model
    actually reads (display_kind is stripped on the wire); a row that still carries
    it must not absorb a human turn even after its kind was dropped.
    """
    from gateway.response_filters import MACHINERY_DISPLAY_KINDS

    if msg.get("display_kind") in MACHINERY_DISPLAY_KINDS:
        return True
    content = msg.get("content")
    if not isinstance(content, str) or "[INTERNAL NOTIFICATION" not in content:
        return False
    from gateway.run_notifications import INTERNAL_NOTIFICATION_FOOTER

    return INTERNAL_NOTIFICATION_FOOTER in content


def _separate_notification_from_user(merged: list, prev: Any, msg: Any) -> bool:
    """Insert an assistant seam when exactly one of two non-empty user rows is a notice.

    An unanswered notice is stored as role=user (a completion note, a wake that drew
    no reply). Folding the next human message onto it places that message under the
    notice's "not written by a human" footer, and the model refuses it. The same weld
    in the other order stamps the footer onto the person's text. Two notices still
    fold below: both are machine text, and providers reject user/user.
    """
    if not _consecutive_plain_users(prev, msg):
        return False
    prev_content, new_content = prev.get("content", ""), msg.get("content", "")
    if not prev_content or not new_content:
        return False
    if _is_gateway_notification(prev) == _is_gateway_notification(msg):
        return False
    merged.append(_notification_boundary())
    merged.append(msg)
    return True


def _account_merged_durable_rows(prev: dict, msg: dict) -> None:
    """Count two persisted rows folded into one survivor; an unpersisted side ends the claim."""
    from agent.context_compressor import _DB_PERSISTED_MARKER
    from agent.conversation_compression_archive import MERGED_DURABLE_ROWS

    if (prev.get(_DB_PERSISTED_MARKER) or prev.get(MERGED_DURABLE_ROWS)) and msg.get(_DB_PERSISTED_MARKER):
        prev[MERGED_DURABLE_ROWS] = int(prev.get(MERGED_DURABLE_ROWS) or 1) + 1
    else:
        prev.pop(MERGED_DURABLE_ROWS, None)


def _release_display_marker_identity(prev: dict, msg: dict) -> None:
    """A display marker folded onto a plain user row must stay addressable (#94486).

    Keeping the marker's display_kind hides the pair from every user-turn index
    (display rows are excluded), so the plain row's id becomes unresolvable and the
    next prompt.submit drops the input. Drop the classification and carry the plain
    row's id. When both rows are markers, keep the first: no plain row is swallowed.
    """
    from agent.context_compressor import _DB_PERSISTED_MARKER

    if not (prev.get("display_kind") and not msg.get("display_kind")):
        return
    marker_row_id = prev.get("_row_id")
    prev.pop("display_kind", None)
    if msg.get("_row_id") is not None:
        if isinstance(marker_row_id, int) and not isinstance(marker_row_id, bool):
            absorbed_ids = prev.setdefault("_absorbed_row_ids", [])
            if marker_row_id not in absorbed_ids:
                absorbed_ids.append(marker_row_id)
        prev["_row_id"] = msg["_row_id"]
    # display_kind is part of the persisted row; reclassifying stales it even when the
    # merged bytes reproduce the persisted content (empty absorb).
    prev.pop(_DB_PERSISTED_MARKER, None)


def _fold_plain_user_pair(prev: Any, msg: Any) -> bool:
    """Weld ``msg`` into ``prev`` when both are consecutive plain human turns. True if folded.

    A /steer row that ended the previous run is already persisted; merging the next
    prompt into it would rewrite it in place and re-break replay parity.
    """
    from agent.prompt_builder import STEER_DISPLAY_KIND

    if not _consecutive_plain_users(prev, msg) or prev.get("display_kind") == STEER_DISPLAY_KIND:
        return False
    from agent.context_compressor import _DB_PERSISTED_MARKER
    from agent.message_metadata import MERGED_TURN_PREFIX
    from agent.turn_context import drop_stale_api_content

    prev_content, new_content = prev.get("content", ""), msg.get("content", "")
    merged_content = (
        (prev_content + "\n\n" + new_content) if prev_content and new_content else (prev_content or new_content)
    )
    had_api_sidecar = "api_content" in prev
    # Read before the marker is popped below. An unpersisted turn folded in ends the claim.
    _account_merged_durable_rows(prev, msg)
    prev["content"] = merged_content
    # The clean-text persist override must replace only the absorbed turn, never the
    # unanswered text before it; kept across replay passes (an empty turn absorbs too).
    if prev_content:
        prev[MERGED_TURN_PREFIX] = prev_content
    drop_stale_api_content(prev)
    _release_display_marker_identity(prev, msg)
    # Pop the persist marker only when the durable row actually changed: a merge that
    # reproduces the persisted bytes (e.g. an empty incoming turn) keeps its stamp.
    if merged_content != prev_content or had_api_sidecar:
        prev.pop(_DB_PERSISTED_MARKER, None)
    from agent.agent_runtime_helpers import _remember_absorbed_row

    _remember_absorbed_row(prev, msg, folded=True)
    return True


def _merge_consecutive_users(messages: list[dict]) -> tuple[list[dict], int]:
    """Pass 3: merge consecutive plain-text user messages, except a gateway notice.

    Human text is never joined onto a notice, in either order: the notice's non-human
    footer would otherwise cover the person's request. A notice still reaches the model
    as its own user turn, with that footer (or its display kind) intact. An ephemeral
    assistant boundary between the two keeps provider role alternation.
    """
    repairs = 0
    merged: list[dict] = []
    for msg in messages:
        prev = merged[-1] if merged and isinstance(merged[-1], dict) else None
        if _separate_notification_from_user(merged, prev, msg):
            repairs += 1
            continue
        if _fold_plain_user_pair(prev, msg):
            repairs += 1
            continue
        merged.append(msg)
    return merged, repairs
