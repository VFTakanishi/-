"""Static sanity checks on the cloud-deploy files (Dockerfile/railway.toml/
.dockerignore) -- these can't be exercised by pytest the way Python code
can (no Docker daemon is assumed to be available here), so this only pins
the specific, previously-audited requirements from CLOUD_DEPLOY.md/the
Dockerfile's own docstring: ffmpeg + a Japanese font are installed, the
app binds 0.0.0.0 and Railway's $PORT, and secrets/large local artifacts
are excluded from the build context.
"""
from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _read(name: str) -> str:
    path = _ROOT / name
    assert path.exists(), f"{name} is missing"
    return path.read_text(encoding="utf-8")


def test_dockerfile_installs_ffmpeg():
    # J
    assert "ffmpeg" in _read("Dockerfile")


def test_dockerfile_installs_japanese_font():
    # K/L: fonts-noto-cjk is the apt package config.py's
    # _LINUX_FONT_PATH_CANDIDATES expects to be installed.
    assert "fonts-noto-cjk" in _read("Dockerfile")


def test_dockerfile_binds_0_0_0_0_and_railway_port():
    content = _read("Dockerfile")
    assert "0.0.0.0" in content
    assert "${PORT" in content


def test_dockerfile_uses_correct_asgi_app_path():
    assert "podcast_clipper.web:app" in _read("Dockerfile")


def test_dockerfile_never_embeds_secrets():
    content = _read("Dockerfile")
    assert "ANTHROPIC_API_KEY=" not in content
    assert "TOOL_PASSWORD=" not in content


def test_dockerignore_excludes_env_and_local_output():
    content = _read(".dockerignore")
    assert ".env" in content
    assert "output/*" in content


def test_railway_toml_points_at_health_endpoint():
    content = _read("railway.toml")
    assert "/health" in content
    assert "Dockerfile" in content
