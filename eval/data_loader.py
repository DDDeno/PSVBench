"""
PSVBench data loading.

Handles QA JSON parsing, video frame sampling,
transcript (subtitle) loading and cleaning, and prompt construction.
"""
from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from eval.frame_sampler import FrameSample, frames_to_data_urls, sample_frames
from eval.qa_schema import QAItem, load_qa_items


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MediaConfig:
    """Video frame loading parameters."""
    num_frames: int = 8
    max_frames: int = 64
    image_max_side: Optional[int] = None
    jpeg_q: Optional[int] = None
    no_video: bool = False


@dataclass(frozen=True)
class TranscriptConfig:
    """Transcript (subtitle) loading parameters."""
    enabled: bool = False
    max_chars: int = 0  # 0 = no truncation


# ---------------------------------------------------------------------------
# Subtitle cleaning
# ---------------------------------------------------------------------------

_VTT_TIMING_RE = re.compile(r"^\s*\d{2}:\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}\.\d{3}")
_SRT_TIMING_RE = re.compile(r"^\s*\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}")
# Also handle short-form VTT timestamps like 00:00.000 --> 00:05.760
_VTT_SHORT_RE = re.compile(r"^\s*\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}\.\d{3}")


def _clean_subtitle_text(raw: str) -> str:
    """Strip VTT/SRT headers, timestamps, and markup; keep only spoken text."""
    lines: List[str] = []
    for line in raw.replace("\ufeff", "").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.upper().startswith("WEBVTT"):
            continue
        if _VTT_TIMING_RE.match(s) or _SRT_TIMING_RE.match(s) or _VTT_SHORT_RE.match(s):
            continue
        if s.isdigit():
            continue
        s = re.sub(r"^<v[^>]*>\s*", "", s).strip()
        s = re.sub(r"</?[^>]+>", "", s).strip()
        if s:
            lines.append(s)
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class DataLoader:
    """Unified loader for QA items, video frames, and transcripts."""

    def __init__(
        self,
        media_cfg: MediaConfig,
        transcript_cfg: TranscriptConfig,
        data_root: Optional[Path] = None,
    ) -> None:
        self.media = media_cfg
        self.transcript = transcript_cfg
        self.data_root = data_root

    def _resolve_path(self, rel_path: str) -> Path:
        """Resolve a relative path from QA data against data_root."""
        p = Path(rel_path)
        if p.is_absolute():
            return p
        if self.data_root is not None:
            return self.data_root / p
        return p

    # -- QA loading ----------------------------------------------------------

    @staticmethod
    def load_qa_items(qa_file: Path) -> List[QAItem]:
        raw = json.loads(qa_file.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(raw, list):
            raise ValueError(f"QA file must be a JSON list: {qa_file}")
        return load_qa_items(raw)

    # -- Frame loading -------------------------------------------------------

    def load_frames(self, item: QAItem) -> Optional[List[str]]:
        """Return base64 data-URL list, or None in no_video mode."""
        if self.media.no_video:
            return None

        video_path = self._resolve_path(item.video_path)
        frames = sample_frames(
            video_path,
            num_frames=self.media.num_frames,
            max_frames=self.media.max_frames,
            max_side=self.media.image_max_side,
            jpeg_q=self.media.jpeg_q,
        )
        return frames_to_data_urls(frames)

    # -- Transcript loading --------------------------------------------------

    def load_transcript(self, item: QAItem) -> Optional[str]:
        """Return cleaned transcript text, or None if not configured / not found."""
        if not self.transcript.enabled:
            return None

        path = self._resolve_path(item.transcript_path)
        if not path.exists():
            return None

        raw = path.read_text(encoding="utf-8", errors="replace")
        text = _clean_subtitle_text(raw)
        if self.transcript.max_chars > 0 and len(text) > self.transcript.max_chars:
            text = text[: self.transcript.max_chars].rstrip() + " …"
        return text or None

    # -- Prompt construction -------------------------------------------------

    def build_prompt(self, item: QAItem, transcript: Optional[str] = None) -> str:
        """Build the final evaluation prompt with optional transcript prefix."""
        opts = item.options

        if self.media.no_video:
            if transcript:
                head = "Answer the multiple-choice question based only on the provided video transcript.\n\n"
            else:
                head = "Answer the multiple-choice question.\n\n"
        elif transcript:
            head = "Answer the multiple-choice question based on the provided video frames and transcript.\n\n"
        else:
            head = "Answer the multiple-choice question based on the provided video frames.\n\n"

        prompt = (
            head
            + f"Question: {item.question}\n\n"
            + f"A. {opts['A']}\nB. {opts['B']}\nC. {opts['C']}\nD. {opts['D']}\n\n"
            + "Reply with only one letter: A, B, C, or D."
        )

        if transcript:
            if self.media.no_video:
                t_head = "You are given the video's transcript (subtitles).\n\n"
            else:
                t_head = "You are also given the video's transcript (subtitles). Use it as additional context.\n\n"
            prompt = t_head + f"Transcript:\n{transcript}\n\n" + prompt

        return prompt
