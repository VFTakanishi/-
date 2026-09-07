import pytest

from podcast_clipper.models import (
    ClipCandidate,
    RawClipCandidate,
    RawUsedSegment,
    TranscriptSegment,
    TranscriptWord,
    UsedSegment,
    find_disfluent_hook_repair_point,
    find_opening_trim_point,
    has_speech_disfluency,
)


def _segment(role="hook", start=0.0, end=1.0):
    return UsedSegment(role=role, start=start, end=end, text="hello")


def test_clip_candidate_accepts_1_to_3_segments():
    for n in (1, 2, 3):
        segs = [_segment(start=i, end=i + 1) for i in range(n)]
        c = ClipCandidate(
            id="c1", hook_type="strong_take", segments=segs, hook_text="h",
            opening_hook_strength=80, title="t", description="d", score=50,
            reasoning="r", caveats="",
        )
        assert len(c.segments) == n


def test_clip_candidate_rejects_zero_or_more_than_3_segments():
    with pytest.raises(ValueError):
        ClipCandidate(
            id="c1", hook_type="strong_take", segments=[], hook_text="h",
            opening_hook_strength=80, title="t", description="d", score=50,
            reasoning="r", caveats="",
        )
    with pytest.raises(ValueError):
        ClipCandidate(
            id="c1", hook_type="strong_take",
            segments=[_segment(start=i, end=i + 1) for i in range(4)],
            hook_text="h", opening_hook_strength=80, title="t", description="d",
            score=50, reasoning="r", caveats="",
        )


def test_clip_candidate_rejects_score_out_of_range():
    with pytest.raises(ValueError):
        ClipCandidate(
            id="c1", hook_type="strong_take", segments=[_segment()],
            hook_text="h", opening_hook_strength=80, title="t", description="d",
            score=101, reasoning="r", caveats="",
        )


def test_clip_candidate_rejects_opening_hook_strength_out_of_range():
    with pytest.raises(ValueError):
        ClipCandidate(
            id="c1", hook_type="strong_take", segments=[_segment()],
            hook_text="h", opening_hook_strength=101, title="t", description="d",
            score=50, reasoning="r", caveats="",
        )
    with pytest.raises(ValueError):
        ClipCandidate(
            id="c1", hook_type="strong_take", segments=[_segment()],
            hook_text="h", opening_hook_strength=-1, title="t", description="d",
            score=50, reasoning="r", caveats="",
        )


def test_used_segment_rejects_inverted_range():
    with pytest.raises(ValueError):
        UsedSegment(role="hook", start=5.0, end=1.0, text="x")


def test_total_duration_sums_segments():
    segs = [_segment(start=0, end=2), _segment(start=10, end=13)]
    c = ClipCandidate(
        id="c1", hook_type="story", segments=segs, hook_text="h",
        opening_hook_strength=80, title="t", description="d", score=10,
        reasoning="r", caveats="",
    )
    assert c.total_duration == 5.0


def test_raw_clip_candidate_segment_count_bounds():
    seg = RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0)
    with pytest.raises(ValueError):
        RawClipCandidate(
            hook_type="open_loop", segments=[], hook_text="h", opening_hook_strength=80,
            title="t", description="d", score=1, reasoning="r", caveats="",
        )
    RawClipCandidate(
        hook_type="open_loop", segments=[seg, seg], hook_text="h", opening_hook_strength=80,
        title="t", description="d", score=1, reasoning="r", caveats="",
    )


def test_raw_clip_candidate_rejects_opening_hook_strength_out_of_range():
    seg = RawUsedSegment(role="hook", start_segment_id=0, end_segment_id=0)
    with pytest.raises(ValueError):
        RawClipCandidate(
            hook_type="open_loop", segments=[seg], hook_text="h", opening_hook_strength=200,
            title="t", description="d", score=1, reasoning="r", caveats="",
        )


def _words(*texts_and_times: tuple[str, float, float]) -> list[TranscriptWord]:
    return [TranscriptWord(start=s, end=e, text=t) for t, s, e in texts_and_times]


def test_find_opening_trim_point_detects_konoyouni_at_start():
    words = _words(
        ("このように", 0.0, 0.55),
        ("弱点を", 0.56, 0.90),
        ("直すと", 0.91, 1.30),
    )
    seg = TranscriptSegment(id=0, start=0.0, end=1.30, text="このように弱点を直すと", words=words)
    trim_word = find_opening_trim_point(seg)
    assert trim_word is not None
    assert trim_word.text == "弱点を"
    assert trim_word.start == 0.56


def test_find_opening_trim_point_detects_other_known_prefix():
    words = _words(("あの", 0.0, 0.2), ("今日", 0.2, 0.5), ("は", 0.5, 0.6))
    seg = TranscriptSegment(id=0, start=0.0, end=0.6, text="あの今日は", words=words)
    trim_word = find_opening_trim_point(seg)
    assert trim_word is not None
    assert trim_word.text == "今日"


def test_find_opening_trim_point_ignores_mid_sentence_occurrence():
    # "このように" only appears once here, but it's not a prefix-buildable
    # match from word 0 ("私" doesn't accumulate toward any known phrase),
    # so no trim point should ever be found -- mirroring the real
    # mid-sentence case ("私はこのように考えています") where the phrase
    # is never at the very start.
    words = _words(
        ("私", 0.0, 0.2),
        ("は", 0.2, 0.3),
        ("このように", 0.3, 0.8),
        ("考えています", 0.8, 1.5),
    )
    seg = TranscriptSegment(id=0, start=0.0, end=1.5, text="私はこのように考えています", words=words)
    assert find_opening_trim_point(seg) is None


def test_find_opening_trim_point_returns_none_without_word_timestamps():
    seg = TranscriptSegment(id=0, start=0.0, end=1.0, text="このように弱点を直すと", words=[])
    assert find_opening_trim_point(seg) is None


def test_find_opening_trim_point_returns_none_when_prefix_is_last_word():
    # A weak lead-in phrase with nothing after it in the segment -- there
    # is no word to move the start to, so no trim point is returned.
    words = _words(("このように", 0.0, 0.55))
    seg = TranscriptSegment(id=0, start=0.0, end=0.55, text="このように", words=words)
    assert find_opening_trim_point(seg) is None


# --- has_speech_disfluency (real-machine incident: word-search / -------
# --- self-correction detection) -----------------------------------------


def test_has_speech_disfluency_detects_word_search_marker():
    marker = has_speech_disfluency(
        "タイヤが丸い状態になるので、ちょっと表現が難しいんですけども、接地面が丸くなるので"
    )
    assert marker is not None
    assert "表現が難しい" in marker


def test_has_speech_disfluency_detects_self_correction_reversal_before_connective():
    # "逆" (reversal marker) appears *before* the connective "っていうのか"
    # -- proximity check must be bidirectional, not just connective-then-
    # marker order.
    marker = has_speech_disfluency("でもこれって、実は逆っていうのか、間違っていて")
    assert marker is not None


def test_has_speech_disfluency_detects_self_correction_reversal_after_connective():
    marker = has_speech_disfluency("燃費改善というか、違ってて")
    assert marker is not None


def test_has_speech_disfluency_detects_standalone_correction_phrase():
    marker = has_speech_disfluency("いや、そうじゃなくて、これはニュートラルの話です。")
    assert marker is not None


def test_has_speech_disfluency_ignores_benign_kanjishi_looking_particle():
    # っていうの"が" (nominalizer) is NOT っていうの"か" (reformulation) --
    # must not be confused by substring overlap.
    assert has_speech_disfluency("ギアを入れてアクセルオフっていうのが、実はニュートラルよりも燃費は良いです") is None


def test_has_speech_disfluency_does_not_reject_plain_reformulation_connective():
    # A bare "というか" used as an ordinary paraphrase connective, with no
    # nearby reversal/negation signal, must never be treated as
    # self-correction on its own.
    assert has_speech_disfluency("燃費改善というか、時短のためにやっています。") is None


def test_has_speech_disfluency_does_not_reject_single_hesitation_filler():
    assert has_speech_disfluency("あの、今日はいい天気ですね。") is None
    assert has_speech_disfluency("まあ、そういうことです。") is None


# --- find_disfluent_hook_repair_point (bounded, word-timestamp-only ----
# --- repair of a disfluent hook preamble) --------------------------------


def _clause_words(*texts: str) -> list[TranscriptWord]:
    """Word list where each string in `texts` is its own single "word"
    (already comma-terminated where relevant) -- sufficient for testing
    clause-boundary detection, which only looks at trailing commas.
    """
    words = []
    t = 0.0
    for text in texts:
        words.append(TranscriptWord(start=t, end=t + 0.5, text=text))
        t += 0.5
    return words


def test_find_disfluent_hook_repair_point_repairs_clean_case():
    # A word-search filler clause immediately followed by an independent,
    # disfluency-free complete sentence -- exactly the "safe to repair"
    # shape (item 6: repair only when the rest of the segment is clean).
    words = _clause_words("ちょっと表現が難しいんですけども、", "要するに新品タイヤは長持ちします。")
    seg = TranscriptSegment(id=0, start=0.0, end=1.0, text="".join(w.text for w in words), words=words)
    repair = find_disfluent_hook_repair_point(seg)
    assert repair is not None
    assert repair.text == "要するに新品タイヤは長持ちします。"


def test_find_disfluent_hook_repair_point_declines_when_disfluency_recurs():
    words = _clause_words(
        "ちょっと表現が難しいんですけども、",
        "こういうことなんですが、",
        "また表現が難しいんですけど、",
        "結局はこういうことです。",
    )
    seg = TranscriptSegment(id=0, start=0.0, end=2.0, text="".join(w.text for w in words), words=words)
    assert find_disfluent_hook_repair_point(seg) is None


def test_find_disfluent_hook_repair_point_declines_candidate3_shaped_hook():
    # The real "candidate 3" incident shape: a disfluent clause
    # ("実は逆っていうのか、") followed by a bare continuative clause
    # ("間違っていて、") that's still grammatically glued to it (not an
    # independent clean clause) -- must never be "repaired" into starting
    # playback mid-correction. This must be rejected outright, not
    # force-rescued.
    words = _clause_words(
        "でもこれって、", "実は逆っていうのか、", "間違っていて、", "一般的な車、", "エンジン車は、",
    )
    seg = TranscriptSegment(id=0, start=0.0, end=2.5, text="".join(w.text for w in words), words=words)
    assert find_disfluent_hook_repair_point(seg) is None


def test_find_disfluent_hook_repair_point_returns_none_without_word_timestamps():
    seg = TranscriptSegment(
        id=0, start=0.0, end=1.0,
        text="ちょっと表現が難しいんですけども、要するにこうです。", words=[],
    )
    assert find_disfluent_hook_repair_point(seg) is None


def test_find_disfluent_hook_repair_point_returns_none_when_nothing_disfluent():
    words = _clause_words("これはとても分かりやすい説明です。")
    seg = TranscriptSegment(id=0, start=0.0, end=0.5, text="".join(w.text for w in words), words=words)
    assert find_disfluent_hook_repair_point(seg) is None
