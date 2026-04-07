"""
Video frame sampling via ffmpeg.

Extracts uniformly-spaced frames from a video and returns them as
base64 data URLs suitable for vision-language model APIs.
"""
from __future__ import annotations

import base64
import dataclasses
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi"}

_SUBPROCESS_TIMEOUT_S = int(os.environ.get("EVAL_SUBPROCESS_TIMEOUT_S", "60"))

# Pull clip_end back by this amount to avoid seeking past the last decodable frame.
_TAIL_MARGIN_S = 0.05


@dataclasses.dataclass(frozen=True)
class FrameSample:
    """A single extracted video frame."""
    image_bytes: bytes
    mime_type: str
    t: float


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------

def _which_ffmpeg() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _ffprobe_duration(video_path: Path) -> Optional[float]:
    exe = shutil.which("ffprobe")
    if exe is None:
        return None
    cmd = [
        exe, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1",
        str(video_path),
    ]
    try:
        proc = subprocess.run(
            cmd, check=False,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=_SUBPROCESS_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def _ffmpeg_duration(video_path: Path) -> Optional[float]:
    """Fallback: parse duration from ffmpeg stderr."""
    import re
    exe = _which_ffmpeg()
    if exe is None:
        return None
    try:
        proc = subprocess.run(
            [exe, "-hide_banner", "-i", str(video_path)],
            check=False,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=_SUBPROCESS_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return None
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", proc.stderr or "")
    if not m:
        return None
    hh, mm, ss = m.groups()
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def _get_duration(video_path: Path) -> Optional[float]:
    return _ffprobe_duration(video_path) or _ffmpeg_duration(video_path)


def _extract_frame(
    video_path: Path,
    t: float,
    *,
    max_side: Optional[int] = None,
    jpeg_q: Optional[int] = None,
) -> Optional[Tuple[bytes, str]]:
    """Extract a single JPEG frame at time *t* seconds."""
    exe = _which_ffmpeg()
    if exe is None:
        return None
    cmd = [
        exe, "-hide_banner", "-loglevel", "error",
        "-ss", f"{max(0.0, t):.3f}",
        "-i", str(video_path),
    ]
    vf: List[str] = []
    if max_side is not None:
        ms = max(8, int(max_side))
        vf.append(f"scale='min(iw,{ms})':'min(ih,{ms})':force_original_aspect_ratio=decrease")
    if vf:
        cmd += ["-vf", ",".join(vf)]
    if jpeg_q is not None:
        cmd += ["-q:v", str(max(2, min(31, int(jpeg_q))))]
    cmd += ["-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "-"]
    try:
        proc = subprocess.run(
            cmd, check=False,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=_SUBPROCESS_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0 or not proc.stdout:
        return None
    return proc.stdout, "image/jpeg"


# ---------------------------------------------------------------------------
# Uniform time selection
# ---------------------------------------------------------------------------

def choose_times(
    clip_start: float,
    clip_end: float,
    num_frames: int = 8,
    max_frames: int = 64,
) -> List[float]:
    """Return uniformly-spaced timestamps (internal-points mode).

    Divides [clip_start, clip_end] into (n+1) equal segments and returns the
    n internal division points.  This avoids landing exactly on the first or
    last frame, which can be black or undecodable.
    """
    if clip_end <= clip_start:
        return [float(clip_start)]
    n = max(1, min(int(num_frames), int(max_frames)))
    if n == 1:
        return [float((clip_start + clip_end) / 2.0)]
    step = (clip_end - clip_start) / (n + 1)
    return [float(clip_start + step * (i + 1)) for i in range(n)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sample_frames(
    video_path: Path,
    *,
    num_frames: int = 8,
    max_frames: int = 64,
    max_side: Optional[int] = None,
    jpeg_q: Optional[int] = None,
) -> List[FrameSample]:
    """Sample *num_frames* uniformly-spaced JPEG frames from a video.

    Args:
        video_path: Path to a video file.
        num_frames: Desired number of frames.
        max_frames: Hard upper bound on frame count.
        max_side: If set, down-scale so the longest side <= this value.
        jpeg_q: ffmpeg JPEG quality (2=best … 31=worst).  None keeps default.

    Returns:
        List of FrameSample with raw JPEG bytes, MIME type, and timestamp.

    Raises:
        FileNotFoundError: Video does not exist.
        ValueError: Unrecognised video extension.
        RuntimeError: Frame extraction failed (ffmpeg missing or broken).
    """
    if video_path.suffix.lower() not in VIDEO_EXTS:
        raise ValueError(f"Not a recognized video extension: {video_path}")
    if not video_path.exists():
        raise FileNotFoundError(str(video_path))

    duration = _get_duration(video_path)
    clip_end = float(duration - _TAIL_MARGIN_S) if duration else 1e9
    clip_end = max(0.0, clip_end)

    times = choose_times(0.0, clip_end, num_frames=num_frames, max_frames=max_frames)

    frames: List[FrameSample] = []
    for t in times:
        result = _extract_frame(video_path, t, max_side=max_side, jpeg_q=jpeg_q)
        if result is None:
            continue
        img_bytes, mime = result
        frames.append(FrameSample(image_bytes=img_bytes, mime_type=mime, t=t))

    if not frames:
        raise RuntimeError(
            "Failed to extract any frames.  Ensure ffmpeg is installed, or run "
            "with --no-video.  Tip: `pip install imageio-ffmpeg` provides a "
            "bundled ffmpeg binary."
        )
    return frames


def frames_to_data_urls(frames: Sequence[FrameSample]) -> List[str]:
    """Convert FrameSamples to base64 data-URL strings."""
    out: List[str] = []
    for f in frames:
        b64 = base64.b64encode(f.image_bytes).decode("ascii")
        out.append(f"data:{f.mime_type};base64,{b64}")
    return out
