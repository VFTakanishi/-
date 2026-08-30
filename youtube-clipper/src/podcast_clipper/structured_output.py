"""Claude Structured Outputs API boundary.

This is the single place the project talks to the Anthropic API. It owns
schema transformation, request construction, stop_reason handling, and
response validation, so callers (clip_selector.py) only build prompts and
convert an already-validated Pydantic model into their own internal
shapes. This module is deliberately schema-agnostic: it knows nothing
about Stage1/Stage2 semantics, only how to get a validated Pydantic
instance out of a Structured Outputs call.

Never uses client.messages.parse(): its post_parser calls
TypeAdapter.validate_json() on the response text unconditionally, before
the caller can inspect stop_reason, so a truncated (stop_reason ==
"max_tokens") or refused (stop_reason == "refusal") response surfaces as
an opaque pydantic.ValidationError instead of a diagnosable stop_reason.
The order here is fixed instead: messages.create() -> check stop_reason
-> extract text -> model_validate_json(). Never repairs malformed JSON
(no regex/bracket-completion/eval/ast.literal_eval/multi-json.loads) and
never retries automatically, at either the application level or the SDK
level (max_retries=0).

Extended thinking is explicitly disabled (thinking={"type": "disabled"}):
a model may default to adaptive thinking when this is omitted, and
thinking tokens count against max_tokens, which can truncate a small
structured-output JSON before it's written even though the JSON itself
is nowhere near the token ceiling.

Schema handling: the raw output of a Pydantic model's model_json_schema()
is never sent to the API directly -- it still contains constraints
Structured Outputs' raw json_schema mode rejects outright (e.g.
`maxItems`, which caused a real 400 on a real machine). Instead this uses
`anthropic.transform_schema`, the SDK's own public, `__all__`-exported
schema transform (the same one client.messages.parse(output_format=...)
uses internally) to build the schema actually sent in
output_config.format.schema. The original Pydantic model is still used
for *local* validation via model_validate_json(), so constraints that
transform_schema strips or folds into descriptive text for the API side
(cf. its handling of `maxItems`) are still enforced locally exactly as
authored -- the API-facing schema and the local validation schema are
different views of the same model, not two independently-maintained
things.
"""
from __future__ import annotations

import anthropic
from pydantic import BaseModel, ValidationError

from . import config


class StructuredOutputError(RuntimeError):
    """A Structured Outputs call did not produce a usable result: it
    stopped before completing (stop_reason != "end_turn") or its
    completed text failed schema validation. Never retried automatically.
    """


def _client() -> anthropic.Anthropic:
    # max_retries=0: a single "解析開始" click must never turn into several
    # hidden API calls -- SDK-level 429/5xx auto-retry is disabled here,
    # same as the no-automatic-retry rule for Stage1/Stage2 logic above it.
    return anthropic.Anthropic(max_retries=0)


def _text_from_response(response) -> str:
    parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
    if not parts:
        types = ", ".join(getattr(b, "type", type(b).__name__) for b in response.content)
        raise StructuredOutputError(f"no text content block in response (content_blocks=[{types}])")
    return "".join(parts)


def call(
    schema_model: type[BaseModel],
    *,
    stage: str,
    system_prompt: str,
    user_content: str,
    max_tokens: int,
):
    """Calls Claude with a Structured Outputs JSON schema derived from
    schema_model via anthropic.transform_schema (never tools/tool_choice,
    never .messages.parse()) and returns a validated instance of
    schema_model. stop_reason is checked, with a distinct diagnosis per
    known reason, before any JSON parsing is attempted.
    """
    response = _client().messages.create(
        model=config.ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
        # Explicitly disabled: a model may default to adaptive extended
        # thinking when this is omitted, which counts against max_tokens
        # and can truncate a small structured-output JSON before it's
        # written -- observed on a real machine as stop_reason=="max_tokens"
        # despite the schema being far smaller than the token ceiling.
        thinking={"type": "disabled"},
        output_config={
            "format": {
                "type": "json_schema",
                "schema": anthropic.transform_schema(schema_model),
            }
        },
    )

    stop_reason = response.stop_reason
    if stop_reason == "max_tokens":
        raise StructuredOutputError(
            f"{stage}: truncated by max_tokens ({max_tokens}) before completing "
            "structured output (stop_reason='max_tokens'). Not retried automatically."
        )
    if stop_reason == "refusal":
        raise StructuredOutputError(
            f"{stage}: Claude refused to produce structured output (stop_reason='refusal')."
        )
    if stop_reason == "model_context_window_exceeded":
        raise StructuredOutputError(
            f"{stage}: model context window exceeded (stop_reason='model_context_window_exceeded')."
        )
    if stop_reason != "end_turn":
        raise StructuredOutputError(
            f"{stage}: unexpected stop_reason={stop_reason!r}; not attempting to parse the response."
        )

    text = _text_from_response(response)
    try:
        return schema_model.model_validate_json(text)
    except ValidationError as exc:
        raise StructuredOutputError(f"{stage}: response text failed schema validation: {exc}") from exc
