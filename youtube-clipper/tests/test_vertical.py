from podcast_clipper import config, vertical


def test_vertical_filter_chain_uses_split_to_avoid_reusing_a_pad_twice():
    filt = vertical.vertical_filter_chain("vcat", "vout")
    # An intermediate pad label can't be referenced twice directly in an
    # ffmpeg filter_complex graph -- it must be split first.
    assert "[vcat]split=2" in filt
    assert filt.endswith("[vout]")


def test_vertical_filter_chain_uses_configured_dimensions(monkeypatch):
    monkeypatch.setattr(config, "VERTICAL_WIDTH", 720)
    monkeypatch.setattr(config, "VERTICAL_HEIGHT", 1280)
    filt = vertical.vertical_filter_chain("in", "out")
    assert "scale=720:1280" in filt
    assert "crop=720:1280" in filt
