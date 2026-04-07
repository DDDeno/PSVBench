#!/usr/bin/env python3
"""
PSVBench evaluation entry point.

Loads QA items, samples video frames (or slides), runs model inference,
and writes per-item predictions + an accuracy summary.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

from eval.data_loader import DataLoader, MediaConfig, TranscriptConfig
from eval.models.base import Prediction
from eval.models.registry import build_model
from eval.qa_schema import QAItem

OOM_EXIT_CODE = 86


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_yaml(path: Path) -> Dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(raw, dict):
        raise ValueError(f"Model config must be a YAML mapping: {path}")
    return raw


def _set_seed(seed: int) -> None:
    """Best-effort determinism: seed Python, numpy, and torch."""
    random.seed(seed)
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


_ANSWER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def _normalize_answer(raw: Optional[str]) -> Optional[str]:
    """Extract a single A/B/C/D letter from model output."""
    if raw is None:
        return None
    s = str(raw).strip().upper()
    if s in {"A", "B", "C", "D"}:
        return s
    m = _ANSWER_RE.search(s)
    return m.group(1).upper() if m else None


def _is_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(p in text for p in (
        "out of memory", "cuda out of memory",
        "cublas_status_alloc_failed", "hip out of memory",
    ))


def _clear_cuda_cache() -> None:
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _read_predictions_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _apply_yaml_eval_defaults(args: argparse.Namespace, model_cfg: dict) -> None:
    """Fill CLI args with defaults from the YAML ``eval:`` section (CLI wins)."""
    defaults = model_cfg.get("eval") or {}
    if not isinstance(defaults, dict):
        return
    mapping = {
        "num_frames": "num_frames",
        "max_frames": "max_frames",
        "image_max_side": "image_max_side",
        "jpeg_q": "jpeg_q",
    }
    for yaml_key, attr in mapping.items():
        cli_val = getattr(args, attr, None)
        yaml_val = defaults.get(yaml_key)
        if cli_val is None and yaml_val is not None:
            setattr(args, attr, yaml_val)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="PSVBench evaluation (per-task + overall accuracy).")

    ap.add_argument("--qa-file", type=Path, required=True, help="QA JSON file path.")
    ap.add_argument("--model-config", type=Path, required=True, help="Model YAML config.")
    ap.add_argument("--output-dir", type=Path, default=None, help="Output directory (default: results/<model>).")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    ap.add_argument("--limit", type=int, default=None, help="Only evaluate first N items (for debugging).")

    # Data root
    ap.add_argument("--data-root", type=Path, default=None,
                    help="Root directory for resolving relative video/transcript paths in QA JSON.")

    # Media options
    ap.add_argument("--num-frames", type=int, default=None, help="Number of frames to sample (default: 8, YAML overridable).")
    ap.add_argument("--max-frames", type=int, default=None, help="Max frame count (default: 64, YAML overridable).")
    ap.add_argument("--image-max-side", type=int, default=None, help="Downscale frames so longest side <= this value.")
    ap.add_argument("--jpeg-q", type=int, default=None, help="ffmpeg JPEG quality: 2 (best) to 31 (worst).")
    ap.add_argument("--no-video", action="store_true", help="Text-only mode (no frames sent to model).")

    # Transcript options
    ap.add_argument("--use-transcript", action="store_true",
                    help="Load and use transcript (subtitle) text from transcript_path in QA data.")

    # Execution options
    ap.add_argument("--resume", action="store_true", help="Resume from existing predictions.jsonl.")
    ap.add_argument("--continue-on-error", action="store_true", help="Record invalid on error and continue.")

    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------------

def _resume_counters(pred_path: Path):
    """Rebuild counters from an existing predictions.jsonl file."""
    correct_task: Counter = Counter()
    total_task: Counter = Counter()
    correct_group: Counter = Counter()
    total_group: Counter = Counter()
    invalid_total = 0
    total_correct = 0
    done_ids: set[str] = set()
    processed = 0

    for r in _read_predictions_jsonl(pred_path):
        rid = r.get("id")
        if rid is not None:
            done_ids.add(str(rid))
        task = str(r.get("task") or "")
        sub_task = str(r.get("sub_task") or "")
        is_correct = bool(r.get("correct", False))
        pred_valid = bool(r.get("pred_valid", False))

        if not pred_valid:
            invalid_total += 1
        total_correct += int(is_correct)

        if task:
            total_group[task] += 1
            correct_group[task] += int(is_correct)
        if sub_task:
            total_task[sub_task] += 1
            correct_task[sub_task] += int(is_correct)
        processed += 1

    return done_ids, processed, total_correct, invalid_total, correct_task, total_task, correct_group, total_group


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _build_summary(
    *,
    model_name: str,
    num_items: int,
    total_correct: int,
    invalid_total: int,
    correct_task: Counter,
    total_task: Counter,
    correct_group: Counter,
    total_group: Counter,
    elapsed: float,
) -> dict:
    per_task = []
    for st, n in sorted(total_task.items(), key=lambda kv: kv[0].lower()):
        c = correct_task[st]
        per_task.append({"sub_task": st, "correct": c, "total": n,
                         "acc": c / n if n else 0.0})

    per_group = []
    for t, n in sorted(total_group.items(), key=lambda kv: kv[0].lower()):
        c = correct_group[t]
        per_group.append({"task": t, "correct": c, "total": n,
                          "acc": c / n if n else 0.0})

    return {
        "model": model_name,
        "num_items": num_items,
        "overall": {
            "correct": total_correct,
            "total": num_items,
            "acc": total_correct / num_items if num_items else 0.0,
            "invalid": invalid_total,
        },
        "per_task_group": per_group,
        "per_task": per_task,
        "elapsed_s": round(elapsed, 3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    model_cfg = _read_yaml(args.model_config)
    _apply_yaml_eval_defaults(args, model_cfg)

    # Fill remaining defaults after YAML merge
    if args.num_frames is None:
        args.num_frames = 8
    if args.max_frames is None:
        args.max_frames = 64

    _set_seed(args.seed)

    loader = DataLoader(
        media_cfg=MediaConfig(
            num_frames=args.num_frames,
            max_frames=args.max_frames,
            image_max_side=args.image_max_side,
            jpeg_q=args.jpeg_q,
            no_video=args.no_video,
        ),
        transcript_cfg=TranscriptConfig(
            enabled=args.use_transcript,
        ),
        data_root=args.data_root,
    )
    model = build_model(model_cfg)
    items = loader.load_qa_items(args.qa_file)

    if args.limit is not None:
        items = items[: args.limit]

    # Output directory
    out_dir = args.output_dir or Path("results") / model.name()
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"

    # Counters
    correct_task: Counter = Counter()
    total_task: Counter = Counter()
    correct_group: Counter = Counter()
    total_group: Counter = Counter()
    invalid_total = 0
    total_correct = 0
    processed = 0
    done_ids: set[str] = set()

    # Resume
    if args.resume and pred_path.exists():
        (done_ids, processed, total_correct, invalid_total,
         correct_task, total_task, correct_group, total_group) = _resume_counters(pred_path)

    items_to_run = [it for it in items if it.id not in done_ids]
    total = len(items)

    # Progress bar
    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=total, initial=processed, dynamic_ncols=True, desc="eval")

    t_start = time.time()
    file_mode = "a" if (args.resume and pred_path.exists()) else "w"

    with pred_path.open(file_mode, encoding="utf-8") as f:
        for it in items_to_run:
            processed += 1
            t0 = time.time()

            # Load media + transcript + prompt
            frames: Optional[List[str]] = None
            pred: Optional[Prediction] = None

            try:
                frames = loader.load_frames(it)
            except Exception as e:
                if _is_oom(e):
                    _clear_cuda_cache()
                if not args.continue_on_error:
                    raise SystemExit(f"Failed to load media for {it.id}: {e}")
                pred = Prediction(answer=None, raw=f"[media_error] {e}",
                                  meta={"exception": type(e).__name__, "stage": "media"})

            if pred is None:
                transcript = loader.load_transcript(it)
                prompt = loader.build_prompt(it, transcript)
                try:
                    pred = model.predict(it, frames_data_urls=frames, prompt=prompt)
                except Exception as e:
                    if _is_oom(e):
                        _clear_cuda_cache()
                    if not args.continue_on_error:
                        raise
                    pred = Prediction(answer=None, raw=f"[predict_error] {e}",
                                      meta={"exception": type(e).__name__, "stage": "predict"})

            dt = time.time() - t0

            # Score
            pred_ans = _normalize_answer(pred.answer)
            pred_valid = pred_ans is not None
            is_correct = pred_valid and pred_ans == it.answer

            if not pred_valid:
                invalid_total += 1
            total_correct += int(is_correct)
            total_task[it.sub_task] += 1
            correct_task[it.sub_task] += int(is_correct)
            total_group[it.task] += 1
            correct_group[it.task] += int(is_correct)

            # Write prediction
            row = {
                "id": it.id,
                "task": it.task,
                "sub_task": it.sub_task,
                "video_path": it.video_path,
                "gold_answer": it.answer,
                "pred_answer": pred_ans,
                "pred_valid": pred_valid,
                "correct": is_correct,
                "latency_s": round(dt, 4),
                "raw": pred.raw,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix_str(f"acc={total_correct/processed:.3f}", refresh=False)

    elapsed = time.time() - t_start
    if pbar is not None:
        pbar.close()

    # Summary
    summary = _build_summary(
        model_name=model.name(),
        num_items=len(items),
        total_correct=total_correct,
        invalid_total=invalid_total,
        correct_task=correct_task,
        total_task=total_task,
        correct_group=correct_group,
        total_group=total_group,
        elapsed=elapsed,
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )

    # Print results
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    for row in summary.get("per_task_group", []):
        print(f"  {row['task']}\t{row['acc']:.3f}\t({row['correct']}/{row['total']})")
    for row in summary.get("per_task", []):
        print(f"  {row['sub_task']}\t{row['acc']:.3f}\t({row['correct']}/{row['total']})")
    print(f"[saved] {out_dir / 'summary.json'}")
    print(f"[saved] {pred_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
