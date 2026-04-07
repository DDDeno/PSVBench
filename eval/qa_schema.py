from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence


@dataclasses.dataclass(frozen=True)
class QAItem:
    id: str
    task: str
    sub_task: str
    video_path: str
    transcript_path: str
    question: str
    options: Dict[str, str]
    answer: str
    gold_window: Optional[Sequence[float]] = None


def _require_str(d: dict, key: str) -> str:
    v = d.get(key)
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"Missing/invalid `{key}` (expected non-empty string).")
    return v.strip()


def _require_options(d: dict) -> Dict[str, str]:
    raw = d.get("options")
    if not isinstance(raw, dict):
        raise ValueError("Missing/invalid `options` (expected object with keys A/B/C/D).")
    opts: Dict[str, str] = {}
    for k in ["A", "B", "C", "D"]:
        v = raw.get(k)
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Missing/invalid `options` (expected non-empty strings for A/B/C/D).")
        opts[k] = v.strip()
    return opts


def _require_answer(d: dict) -> str:
    ans = d.get("answer")
    if not isinstance(ans, str):
        raise ValueError("Missing/invalid `answer` (expected string A/B/C/D).")
    ans = ans.strip().upper()
    if ans not in {"A", "B", "C", "D"}:
        raise ValueError("Missing/invalid `answer` (expected A/B/C/D).")
    return ans


def load_qa_items(items: List[dict]) -> List[QAItem]:
    out: List[QAItem] = []
    for i, d in enumerate(items):
        if not isinstance(d, dict):
            raise ValueError(f"QA item at index {i} is not an object.")
        qa_id = _require_str(d, "id") if isinstance(d.get("id"), str) else f"item_{i:06d}"
        task = _require_str(d, "task")
        sub_task = _require_str(d, "sub_task")
        video_path = _require_str(d, "video_path")
        transcript_path = _require_str(d, "transcript_path")
        question = _require_str(d, "question")
        options = _require_options(d)
        answer = _require_answer(d)
        gold_window = d.get("gold_window")
        if gold_window is not None:
            if (
                not isinstance(gold_window, (list, tuple))
                or len(gold_window) != 2
                or not all(isinstance(x, (int, float)) for x in gold_window)
            ):
                raise ValueError("Invalid `gold_window` (expected [start, end] floats).")
        out.append(
            QAItem(
                id=qa_id,
                task=task,
                sub_task=sub_task,
                video_path=video_path,
                transcript_path=transcript_path,
                question=question,
                options=options,
                answer=answer,
                gold_window=gold_window,
            )
        )
    return out
