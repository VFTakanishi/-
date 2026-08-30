"""Contract tests for the Claude Structured Outputs API boundary
(structured_output.py). Two layers, per the API-boundary audit:

1. Fake-response unit tests: verify our own branching logic (stop_reason
   checked before any JSON parsing, no repair, no retry) against plain
   Python fakes -- fast, but they say nothing about what the SDK actually
   puts on the wire.
2. httpx2.MockTransport contract tests: run a *real* anthropic.Anthropic
   client through its actual request-building/serialization code, with
   httpx2.MockTransport intercepting at the transport layer before any
   socket opens (a structural, not conventional, zero-network guarantee --
   verified against the installed anthropic==1.0.0 SDK). These are what
   would have caught the real "maxItems unsupported" 400: they inspect the
   literal request body, not just the Python kwargs we thought we passed.

No real Anthropic API call, no external HTTP, in either layer.
"""
import json

import anthropic
import httpx2
import pytest
from pydantic import ValidationError

from podcast_clipper import clip_selector, structured_output
from podcast_clipper.clip_selector import Stage1Output, Stage2RankingOutput


# --- Layer 1: fake-response unit tests -----------------------------------


class _FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, *, stop_reason, content):
        self.stop_reason = stop_reason
        self.content = content


class _FakeMessages:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class _FakeClient:
    def __init__(self, response):
        self.messages = _FakeMessages(response)


def test_call_returns_parsed_model_on_end_turn(monkeypatch):
    valid_json = Stage1Output(candidates=[]).model_dump_json()
    response = _FakeResponse(stop_reason="end_turn", content=[_FakeTextBlock(valid_json)])
    monkeypatch.setattr(structured_output, "_client", lambda: _FakeClient(response))

    result = structured_output.call(
        Stage1Output, stage="Stage1", system_prompt="sys", user_content="user", max_tokens=100
    )
    assert result.candidates == []


def test_call_raises_without_parsing_on_max_tokens(monkeypatch):
    # stop_reason != "end_turn" -> must fail before ever attempting to
    # parse JSON. Feed clearly-truncated JSON to prove parsing never runs
    # (a parse attempt would raise a ValidationError, not this message).
    truncated_json = '{"candidates": [{"hook_type": "surprising_fact"'
    response = _FakeResponse(stop_reason="max_tokens", content=[_FakeTextBlock(truncated_json)])
    monkeypatch.setattr(structured_output, "_client", lambda: _FakeClient(response))

    with pytest.raises(structured_output.StructuredOutputError, match="max_tokens"):
        structured_output.call(
            Stage1Output, stage="Stage1", system_prompt="sys", user_content="user", max_tokens=100
        )


def test_call_raises_without_parsing_on_refusal(monkeypatch):
    response = _FakeResponse(stop_reason="refusal", content=[_FakeTextBlock("I can't help with that")])
    monkeypatch.setattr(structured_output, "_client", lambda: _FakeClient(response))

    with pytest.raises(structured_output.StructuredOutputError, match="refusal"):
        structured_output.call(
            Stage1Output, stage="Stage1", system_prompt="sys", user_content="user", max_tokens=100
        )


def test_call_raises_without_parsing_on_model_context_window_exceeded(monkeypatch):
    response = _FakeResponse(stop_reason="model_context_window_exceeded", content=[])
    monkeypatch.setattr(structured_output, "_client", lambda: _FakeClient(response))

    with pytest.raises(structured_output.StructuredOutputError, match="context window"):
        structured_output.call(
            Stage1Output, stage="Stage1", system_prompt="sys", user_content="user", max_tokens=100
        )


def test_call_raises_on_unexpected_stop_reason(monkeypatch):
    response = _FakeResponse(stop_reason="some_future_reason", content=[_FakeTextBlock("{}")])
    monkeypatch.setattr(structured_output, "_client", lambda: _FakeClient(response))

    with pytest.raises(structured_output.StructuredOutputError, match="some_future_reason"):
        structured_output.call(
            Stage1Output, stage="Stage1", system_prompt="sys", user_content="user", max_tokens=100
        )


def test_call_raises_on_invalid_json_without_repair(monkeypatch):
    # stop_reason == "end_turn" but the completed text isn't valid JSON for
    # the schema -- must fail immediately, never attempt any repair.
    truncated_json = '{"candidates": [{"hook_type": "surprising_fact"'
    response = _FakeResponse(stop_reason="end_turn", content=[_FakeTextBlock(truncated_json)])
    monkeypatch.setattr(structured_output, "_client", lambda: _FakeClient(response))

    with pytest.raises(structured_output.StructuredOutputError, match="schema validation"):
        structured_output.call(
            Stage1Output, stage="Stage1", system_prompt="sys", user_content="user", max_tokens=100
        )


def test_call_raises_when_no_text_block_present(monkeypatch):
    response = _FakeResponse(stop_reason="end_turn", content=[])
    monkeypatch.setattr(structured_output, "_client", lambda: _FakeClient(response))

    with pytest.raises(structured_output.StructuredOutputError):
        structured_output.call(
            Stage1Output, stage="Stage1", system_prompt="sys", user_content="user", max_tokens=100
        )


def test_client_disables_sdk_level_retries(monkeypatch):
    captured = {}

    def _spy(**kwargs):
        captured.update(kwargs)
        return _FakeClient(_FakeResponse(stop_reason="end_turn", content=[]))

    monkeypatch.setattr(structured_output.anthropic, "Anthropic", _spy)
    structured_output._client()
    assert captured.get("max_retries") == 0


def test_forbid_real_anthropic_client_used_by_other_modules(monkeypatch):
    # Mirrors clip_selector's own safety fixture: confirms poisoning
    # structured_output.anthropic.Anthropic actually blocks construction
    # (this is what tests/test_clip_selector.py relies on).
    def _forbidden(*args, **kwargs):
        raise AssertionError("real anthropic.Anthropic() must not be instantiated in tests")

    monkeypatch.setattr(structured_output.anthropic, "Anthropic", _forbidden)
    with pytest.raises(AssertionError):
        structured_output._client()


# --- Layer 2: httpx2.MockTransport contract tests (real SDK, zero network) -


def _client_with_mock_transport(handler):
    """A *real* anthropic.Anthropic client whose http_client is a
    httpx2.Client backed by httpx2.MockTransport. MockTransport intercepts
    at the transport layer before any socket is opened -- this is a
    structural (not just conventional) zero-network guarantee. Every real
    anthropic.Anthropic() construction in this file MUST go through this
    helper; a bare anthropic.Anthropic() call here would be a bug.
    """
    return anthropic.Anthropic(
        api_key="sk-test-not-real",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )


def _message_response(*, text: str, stop_reason: str = "end_turn") -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": text}],
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 10},
        },
    )


def _walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def _assert_no_unsupported_constraint_keys(schema: dict):
    """Structured Outputs' raw json_schema mode rejects constraints like
    `maxItems` outright (the real 400 this task investigates). This walks
    every object in the (already anthropic.transform_schema()-processed)
    schema and asserts none of them carry these keys literally -- the
    information may still appear as human-readable text inside a
    `description` string (transform_schema's documented behavior for
    `maxItems`/most `minimum`/`maximum`), which is fine; it's the raw
    JSON-Schema keyword landing in the request that caused the 400.
    """
    forbidden = {"maxItems", "minimum", "maximum"}
    for node in _walk(schema):
        leaked = forbidden & node.keys()
        assert not leaked, f"unsupported constraint key(s) {leaked} leaked into request schema: {node}"


def test_stage1_request_body_has_no_unsupported_constraint_keys():
    captured = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["request"] = request
        return _message_response(text=json.dumps({"candidates": []}))

    client = _client_with_mock_transport(handler)
    real_client = structured_output._client
    structured_output._client = lambda: client
    try:
        result = structured_output.call(
            Stage1Output, stage="Stage1", system_prompt="sys prompt",
            user_content="user content", max_tokens=2048,
        )
    finally:
        structured_output._client = real_client

    assert result.candidates == []

    body = json.loads(captured["request"].content)
    assert body["model"]
    assert body["max_tokens"] == 2048
    assert body["system"] == "sys prompt"
    assert body["messages"] == [{"role": "user", "content": "user content"}]
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "tools" not in body
    assert "tool_choice" not in body
    assert "thinking" not in body
    assert body.get("stream") in (None, False)

    schema = body["output_config"]["format"]["schema"]
    _assert_no_unsupported_constraint_keys(schema)
    # additionalProperties: false is forced by transform_schema on every
    # object -- confirm it's present at the top level and inside $defs.
    assert schema.get("additionalProperties") is False
    for name, sub in schema.get("$defs", {}).items():
        if sub.get("type") == "object":
            assert sub.get("additionalProperties") is False, f"$defs.{name} missing additionalProperties:false"


def test_stage2_request_body_has_no_unsupported_constraint_keys_and_no_tools():
    captured = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["request"] = request
        return _message_response(text=json.dumps({"ranked_candidate_ids": []}))

    client = _client_with_mock_transport(handler)
    real_client = structured_output._client
    structured_output._client = lambda: client
    try:
        result = structured_output.call(
            Stage2RankingOutput, stage="Stage2", system_prompt="sys prompt",
            user_content="user content", max_tokens=512,
        )
    finally:
        structured_output._client = real_client

    assert result.ranked_candidate_ids == []

    body = json.loads(captured["request"].content)
    assert body["max_tokens"] == 512
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "tools" not in body
    assert "tool_choice" not in body

    schema = body["output_config"]["format"]["schema"]
    _assert_no_unsupported_constraint_keys(schema)
    assert schema.get("additionalProperties") is False


def test_mock_transport_stop_reason_max_tokens_raises_before_parsing():
    # End-to-end through the real SDK: a stop_reason=max_tokens response
    # (with deliberately-truncated JSON text) must raise StructuredOutputError
    # without ever reaching model_validate_json.
    def handler(request: httpx2.Request) -> httpx2.Response:
        return _message_response(
            text='{"candidates": [{"hook_type": "story"', stop_reason="max_tokens"
        )

    client = _client_with_mock_transport(handler)
    real_client = structured_output._client
    structured_output._client = lambda: client
    try:
        with pytest.raises(structured_output.StructuredOutputError, match="max_tokens"):
            structured_output.call(
                Stage1Output, stage="Stage1", system_prompt="sys",
                user_content="user", max_tokens=100,
            )
    finally:
        structured_output._client = real_client


# --- Pseudo API-boundary E2E: fixture transcript -> chunk -> Stage1 ------
# --- MockTransport responses -> chunk cache -> local filter -> Stage2 ----
# --- MockTransport response -> 3 final candidates ------------------------


def _fixture_transcript(video_id="e2e-vid"):
    from podcast_clipper.models import Transcript, TranscriptSegment, TranscriptWord

    segments = []
    for i in range(3):
        start = i * 20.0
        text = f"強い発言その{i}です、続きが気になる内容です"
        segments.append(
            TranscriptSegment(
                id=i, start=start, end=start + 15.0, text=text,
                words=[TranscriptWord(start=start, end=start + 15.0, text=text)],
            )
        )
    return Transcript(video_id=video_id, language="ja", segments=segments)


def test_pseudo_api_boundary_e2e_reaches_three_final_candidates(monkeypatch):
    from podcast_clipper import config

    transcript = _fixture_transcript()
    monkeypatch.setattr(config, "DURATION_HARD_MIN_SEC", 0.0)
    monkeypatch.setattr(config, "DURATION_HARD_MAX_SEC", 100.0)
    monkeypatch.setattr(config, "MIN_OPENING_HOOK_STRENGTH", 60)
    monkeypatch.setattr(config, "CHUNK_MINUTES", 100.0)  # single chunk
    monkeypatch.setattr(config, "CHUNK_OVERLAP_MINUTES", 0.0)

    stage1_payload = json.dumps({
        "candidates": [
            {
                "hook_type": "story",
                "segments": [{"role": "hook", "start_segment_id": i, "end_segment_id": i}],
                "opening_hook_strength": 90,
                "score": 80,
            }
            for i in range(3)
        ]
    })

    call_log = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        schema_title = body["output_config"]["format"]["schema"].get("title", "")
        call_log.append(schema_title)
        if schema_title == "Stage1Output":
            return _message_response(text=stage1_payload)
        # Stage2: rank all 3 candidate ids.
        return _message_response(
            text=json.dumps({"ranked_candidate_ids": [f"s1_c{i:03d}" for i in range(3)]})
        )

    client = _client_with_mock_transport(handler)
    real_client = structured_output._client
    structured_output._client = lambda: client
    try:
        result = clip_selector.select_candidates(transcript, "タイトル", force_refresh=True)
    finally:
        structured_output._client = real_client

    assert len(result) == 3
    assert call_log == ["Stage1Output", "Stage2RankingOutput"]
