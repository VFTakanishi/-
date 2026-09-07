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
    find_speech_restart_marker,
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


# --- find_speech_restart_marker (real-machine incident: an unfinished ---
# --- clause abandoned and restarted with the same content phrase) -------


def test_find_speech_restart_marker_detects_candidate1_real_example():
    # A: the real incident -- "Nレンジで下るというのは、" is left dangling on
    # "は、" (never completing a predicate), and the very next clause
    # restarts with the same content phrase ("Nレンジ") in a different
    # construction ("にすると"). has_speech_disfluency alone cannot catch
    # this (no word-search marker, no reformulation/reversal connective).
    words = _clause_words(
        "ホンダも取扱説明書の中に、",
        "走行中にNレンジで下るというのは、",
        "Nレンジにすると、",
        "エンジンブレーキが効かなくなって、",
        "思わぬ事故の原因になるっていうふうに明記もしているので、",
    )
    marker = find_speech_restart_marker(words)
    assert marker is not None
    assert "Nレンジ" in marker


def test_find_speech_restart_marker_ignores_natural_emphasis_repetition():
    # B: repeating a claim for emphasis ("燃費が良い、燃費が良いというだけで
    # なく安全性も高い") -- the first clause ends on "い、", not a bare
    # topic/case particle, so it is never treated as a dangling clause.
    words = _clause_words(
        "燃費が良い、",
        "燃費が良いというだけではなく安全性も高い。",
    )
    assert find_speech_restart_marker(words) is None


def test_find_speech_restart_marker_ignores_hook_payoff_exact_repeat_structure():
    # C: the existing, intentionally-allowed hook->context->payoff exact
    # repeat feature reuses a whole resolved segment verbatim, and hook
    # and payoff are always separate RawUsedSegments checked independently
    # (never concatenated into one word list -- see
    # clip_selector._candidate_speech_restart_marker). Even in the
    # degenerate case of two complete, comma-free sentences placed back to
    # back, this function must not flag anything: neither sentence ends on
    # a bare topic/case particle, so no clause here is ever a restart
    # candidate.
    words = _clause_words(
        "冷却効率を上げるために重量を増やすというのはアンチパターンになると思います。",
        "冷却効率を上げるために重量を増やすというのはアンチパターンになると思います。",
    )
    assert find_speech_restart_marker(words) is None


def test_find_speech_restart_marker_ignores_natural_enumeration():
    # D: a natural list of model names ("86、BRZ、GR86") -- each item ends
    # on a bare noun before the comma, never on a topic/case particle, so
    # it is never treated as a dangling clause in the first place.
    words = _clause_words("86、", "BRZ、", "GR86について話します。")
    assert find_speech_restart_marker(words) is None


def test_find_speech_restart_marker_ignores_legitimate_parallel_construction():
    # A dangling-looking clause ("彼は学校に、") followed by an unrelated
    # independent clause sharing no real content phrase (only the
    # single-character particle "は" overlaps, below the minimum shared
    # content-phrase length) -- this is ordinary parallel/ellipsis
    # Japanese, not a restart, and must not be flagged.
    words = _clause_words("彼は学校に、", "私は家に帰った。")
    assert find_speech_restart_marker(words) is None


def test_find_speech_restart_marker_returns_none_without_word_timestamps():
    # F: no word-timestamp data at all -- must never guess-reject via this
    # mechanism.
    assert find_speech_restart_marker([]) is None


def test_find_speech_restart_marker_skips_clause_already_flagged_by_disfluency():
    # A dangling-ending clause that already contains a speech_disfluency
    # marker ("ちょっと表現が難しいんですけども、" ends on "も、") must not
    # also be flagged as a restart -- the two gates must not double-signal
    # on the same underlying text.
    words = _clause_words(
        "ちょっと表現が難しいんですけども、",
        "要するに新品タイヤは長持ちします。",
    )
    assert find_speech_restart_marker(words) is None


def test_find_speech_restart_marker_detects_restart_split_across_segments():
    # The restart pattern split across two adjacent original transcript
    # segments (as faster-whisper's VAD might do) -- the caller
    # concatenates both segments' word lists before calling this function,
    # so the restart must still be caught.
    segment1_words = _clause_words(
        "ホンダも取扱説明書の中に、",
        "走行中にNレンジで下るというのは、",
    )
    segment2_words = _clause_words(
        "Nレンジにすると、",
        "エンジンブレーキが効かなくなって、",
    )
    marker = find_speech_restart_marker(segment1_words + segment2_words)
    assert marker is not None
    assert "Nレンジ" in marker


def test_find_speech_restart_marker_detects_restart_inside_body_not_only_hook():
    # G: a restart occurring mid-body (not at the very start of the word
    # list) must still be detected -- the function scans every adjacent
    # clause pair, not just the first one.
    words = _clause_words(
        "この動画では車の話をします、",
        "燃費については、",
        "燃費の話をすると、",
        "実は改善の余地があります。",
    )
    marker = find_speech_restart_marker(words)
    assert marker is not None
    assert "燃費" in marker
