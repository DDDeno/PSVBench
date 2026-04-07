from __future__ import annotations

import base64
import io
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from eval.models.base import ModelAdapter, Prediction
from eval.qa_schema import QAItem


_ANS_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def _parse_answer_letter(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.strip().upper()
    if t in {"A", "B", "C", "D"}:
        return t
    m = _ANS_RE.search(t)
    if not m:
        return None
    return m.group(1).upper()


def _decode_data_url_image(data_url: str) -> Tuple[bytes, str]:
    if not data_url.startswith("data:"):
        raise ValueError("Expected a data URL (data:<mime>;base64,...)")
    try:
        header, b64 = data_url.split(",", 1)
        mime = header.split(";", 1)[0].split(":", 1)[1]
        raw = base64.b64decode(b64.encode("ascii"), validate=False)
        return raw, mime
    except Exception as e:
        raise ValueError(f"Invalid data URL: {e}") from e


def _ensure_attn_implementation(model: Any, *, default: str = "sdpa") -> None:
    """
    transformers>=4.50 routes attention by `config._attn_implementation`.
    Some trust_remote_code vision backbones (e.g., SigLIP inside mPLUG-Owl3) may leave it as None,
    causing KeyError: None at runtime.
    """
    # Keep the import local to avoid paying cost when not needed.
    try:
        from transformers.modeling_utils import PreTrainedModel  # type: ignore
    except Exception:
        PreTrainedModel = None  # type: ignore

    # Pick a conservative fallback.
    impl = str(default).strip() or "sdpa"
    if impl not in {"sdpa", "eager", "flash_attention_2"}:
        impl = "sdpa"

    # Walk submodules and fix configs in-place (best-effort).
    for m in getattr(model, "modules", lambda: [])():
        cfg = getattr(m, "config", None)
        if cfg is None:
            continue
        try:
            cur = getattr(cfg, "_attn_implementation", "MISSING")
        except Exception:
            continue
        if cur is None:
            try:
                setattr(cfg, "_attn_implementation", impl)
            except Exception:
                pass


def _normalize_device_spec(device: Any) -> Optional[str]:
    if device is None:
        return None
    if isinstance(device, int):
        return f"cuda:{device}"
    text = str(device).strip()
    if not text or text in {"cpu", "disk"}:
        return None
    if text == "cuda":
        return "cuda:0"
    return text


def _select_input_device(model: Any, *, fallback: str) -> str:
    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, dict):
        return fallback

    for _module_name, target in device_map.items():
        normalized = _normalize_device_spec(target)
        if normalized is not None:
            return normalized
    return fallback


@dataclass(frozen=True)
class MPlugOwl3Config:
    name: str
    model_path: str
    device: str = "auto"  # auto|cpu|cuda
    torch_dtype: str = "auto"  # auto|float32|float16|bfloat16
    trust_remote_code: bool = True
    attn_implementation: Optional[str] = None
    max_new_tokens: int = 16
    temperature: float = 0.0
    top_p: float = 1.0


class MPlugOwl3Adapter(ModelAdapter):
    def __init__(self, cfg: MPlugOwl3Config) -> None:
        self._cfg = cfg

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if cfg.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = cfg.device
        self._device = device
        self._input_device = "cpu"

        if cfg.torch_dtype == "auto":
            resolved_torch_dtype = None
        elif cfg.torch_dtype == "float32":
            resolved_torch_dtype = torch.float32
        elif cfg.torch_dtype == "float16":
            resolved_torch_dtype = torch.float16
        elif cfg.torch_dtype == "bfloat16":
            resolved_torch_dtype = torch.bfloat16
        else:
            raise ValueError(f"Unsupported torch_dtype: {cfg.torch_dtype}")

        self._tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=bool(cfg.trust_remote_code))

        model_kwargs: Dict[str, Any] = {"trust_remote_code": bool(cfg.trust_remote_code), "low_cpu_mem_usage": True}
        if resolved_torch_dtype is not None:
            model_kwargs["torch_dtype"] = resolved_torch_dtype
        if cfg.attn_implementation:
            model_kwargs["attn_implementation"] = str(cfg.attn_implementation)
        if device == "cuda":
            model_kwargs["device_map"] = "balanced" if torch.cuda.device_count() > 1 else "auto"

        self._model = AutoModelForCausalLM.from_pretrained(cfg.model_path, **model_kwargs)
        if device != "cuda":
            self._model.to(device)
        self._model.eval()
        _ensure_attn_implementation(self._model, default=str(cfg.attn_implementation or "sdpa"))
        self._input_device = _select_input_device(self._model, fallback="cuda:0" if device == "cuda" else device)

        if not hasattr(self._model, "init_processor"):
            raise RuntimeError("mPLUG-Owl3 model missing `init_processor(tokenizer)`; check model_path.")
        self._processor = self._model.init_processor(self._tokenizer)

    def name(self) -> str:
        return self._cfg.name

    def predict(
        self,
        item: QAItem,
        *,
        frames_data_urls: Optional[Sequence[str]],
        prompt: str,
    ) -> Prediction:
        import torch
        from PIL import Image

        frames: Sequence[Image.Image] = []
        if frames_data_urls:
            decoded = []
            for u in frames_data_urls:
                raw, _mime = _decode_data_url_image(str(u))
                decoded.append(Image.open(io.BytesIO(raw)).convert("RGB"))
            frames = decoded

        system_msg = (
            "You are a strict multiple-choice evaluator. "
            "Reply with only a single letter: A, B, C, or D."
        )
        user_text = f"<|video|>\n{prompt}"
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": ""},
        ]

        video_frames = [list(frames)] if frames else None

        with torch.inference_mode():
            inputs = self._processor(messages, videos=video_frames, return_tensors="pt")
            if hasattr(inputs, "to"):
                inputs = inputs.to(self._input_device)
            else:
                for k, v in list(inputs.items()):
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self._input_device)

            if "pixel_values" in inputs and isinstance(inputs["pixel_values"], torch.Tensor):
                try:
                    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=next(self._model.parameters()).dtype)
                except Exception:
                    pass

            gen_kwargs: Dict[str, Any] = {"max_new_tokens": int(self._cfg.max_new_tokens)}
            if float(self._cfg.temperature) > 0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = float(self._cfg.temperature)
                gen_kwargs["top_p"] = float(self._cfg.top_p)
            else:
                gen_kwargs["do_sample"] = False

            out = self._model.generate(
                **inputs,
                tokenizer=self._tokenizer,
                decode_text=True,
                **gen_kwargs,
            )

        if isinstance(out, (list, tuple)) and out:
            text = str(out[0])
        else:
            text = str(out)

        ans = _parse_answer_letter(text)
        return Prediction(
            answer=ans,
            raw=text,
            meta={
                "qa_id": item.id,
                "model_path": self._cfg.model_path,
                "device": self._device,
                "input_device": self._input_device,
                "torch_dtype": self._cfg.torch_dtype,
                "transformers_cache": os.environ.get("TRANSFORMERS_CACHE") or os.environ.get("HF_HOME") or "",
            },
        )


def build_from_config(cfg: dict) -> MPlugOwl3Adapter:
    name = str(cfg.get("name", "mplug_owl3")).strip()
    model_path = str(cfg.get("model_path", "")).strip()
    if not model_path:
        raise ValueError("Missing `model_path` in model config.")
    return MPlugOwl3Adapter(
        MPlugOwl3Config(
            name=name,
            model_path=model_path,
            device=str(cfg.get("device", "auto")).strip() or "auto",
            torch_dtype=str(cfg.get("torch_dtype", "auto")).strip() or "auto",
            trust_remote_code=bool(cfg.get("trust_remote_code", True)),
            attn_implementation=str(cfg.get("attn_implementation", "")).strip() or None,
            max_new_tokens=int(cfg.get("max_new_tokens", 16)),
            temperature=float(cfg.get("temperature", 0.0)),
            top_p=float(cfg.get("top_p", 1.0)),
        )
    )
