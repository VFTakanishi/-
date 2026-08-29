"""Internal debug CLI (absolute condition #3: the browser UI is the main
entry point; this exists only for scripting/debugging the pipeline
directly, synchronously, without going through jobs.py).
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import click

from . import boundary, cache, clip_selector, config, ingest, qa, render, transcribe


@click.group()
def cli() -> None:
    pass


@cli.command(name="analyze-file")
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option("--force-refresh", is_flag=True, default=False)
def analyze_file(path: str, force_refresh: bool) -> None:
    """Ingest a local video file, transcribe, and select 3 candidates. Prints JSON."""
    source = Path(path)
    with open(source, "rb") as fileobj:
        result = ingest.ingest_uploaded_file(fileobj, source.name)

    transcript = transcribe.transcribe_video(
        result.path, result.video_id, force_refresh=force_refresh
    )
    raw_candidates = clip_selector.select_candidates(
        transcript, result.title, force_refresh=force_refresh
    )
    resolved = [
        boundary.resolve_candidate(rc, transcript, candidate_id=f"c{i + 1}")
        for i, rc in enumerate(raw_candidates)
    ]
    click.echo(
        json.dumps(
            {
                "video_id": result.video_id,
                "video_title": result.title,
                "candidates": [asdict(c) for c in resolved],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@cli.command()
@click.argument("video_id")
@click.argument("candidate_id")
def render_cmd(video_id: str, candidate_id: str) -> None:
    """Render + QA one cached candidate_id (e.g. c1/c2/c3) for video_id."""
    transcript = cache.load_transcript(video_id)
    raw_candidates = cache.load_stage2(video_id)
    if transcript is None or raw_candidates is None:
        raise click.ClickException(f"no cache found for video_id={video_id}; run analyze first")

    idx = int(candidate_id.lstrip("c")) - 1
    raw_candidate = raw_candidates[idx]
    resolved_candidate = boundary.resolve_candidate(raw_candidate, transcript, candidate_id)

    source_files = list((config.OUTPUT_DIR / video_id / "source").glob(f"{video_id}.*"))
    if not source_files:
        raise click.ClickException(f"source video not found under output/{video_id}/source")

    manifest = render.render_candidate(source_files[0], resolved_candidate, video_id)
    qa_report = qa.run_full_qa(raw_candidate, transcript, manifest, source_files[0])

    click.echo(
        json.dumps(
            {
                "manifest": asdict(manifest),
                "qa_checks": [asdict(c) for c in qa_report.checks],
                "thumbnails": qa_report.thumbnails,
                "download_allowed": qa_report.download_allowed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


cli.add_command(render_cmd, name="render")


if __name__ == "__main__":
    cli()
