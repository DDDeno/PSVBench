from __future__ import annotations

import base64
import io
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from eval.models.base import ModelAdapter, Prediction
from eval.qa_schema import QAItem

try:
    from flexselect.modeling_qwen2_5_vl import (
        Qwen2_5_VLConfig,
        Qwen2_5_VLForConditionalGeneration,
    )
    from flexselect.processing_qwen2_5_vl import Qwen2_5_VLProcessor
    from qwen_vl_utils import process_vision_info
except ImportError as e:
    raise ImportError(
        "FlexSelect requires external installation. Please install it:\n"
        "  git clone https://github.com/yunzhuzhang0918/flexselect\n"
        "  cd flexselect && pip install -e .\n"
        "  pip install qwen-vl-utils"
    ) from e


_ANS_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def _parse_answer_letter(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.strip().upper()
    if t in {"A", "B", "C", "D"}:
        return t
    m = _ANS_RE.search(t)
    return m.group(1).upper() if m else None


def _decode_data_url_image(data_url: str) -> bytes:
    if not data_url.startswith("data:"):
        raise ValueError("Expected a data URL (data:<mime>;base64,...)")
    header, b64 = data_url.split(",", 1)
    if ";base64" not in header:
        raise ValueError("Expected base64 data URL")
    return base64.b64decode(b64.encode("ascii"), validate=False)


@dataclass(frozen=True)
class FlexSelectQwen25VLConfig:
    name: str
    base_model_path: str
    token_selector_path: str
    use_token_selector: bool = True
    token_selector_layer: int = -1
    drop_func_name: str = "token_selection"
    tkn_budget: int = 7040
    device: str = "auto"  # auto|cpu|cuda
    torch_dtype: str = "auto"  # auto|float32|float16|bfloat16
    attn_implementation: Optional[str] = None
    max_new_tokens: int = 16
    temperature: float = 0.0
    top_p: float = 1.0
    min_pixels: int = 3136
    max_pixels: int = 16384 * 28 * 28


class FlexSelectQwen25VLAdapter(ModelAdapter):
    def __init__(self, cfg: FlexSelectQwen25VLConfig) -> None:
        self._cfg = cfg

        import torch

        if cfg.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = cfg.device
        self._device = device

        if cfg.torch_dtype == "auto":
            resolved_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        elif cfg.torch_dtype == "float32":
            resolved_dtype = torch.float32
        elif cfg.torch_dtype == "float16":
            resolved_dtype = torch.float16
        elif cfg.torch_dtype == "bfloat16":
            resolved_dtype = torch.bfloat16
        else:
            raise ValueError(f"Unsupported torch_dtype: {cfg.torch_dtype}")

        model_cfg = Qwen2_5_VLConfig.from_pretrained(cfg.base_model_path)
        if bool(cfg.use_token_selector):
            setattr(model_cfg, "use_token_selector", True)
            setattr(model_cfg, "token_selector_path", str(cfg.token_selector_path))
            setattr(model_cfg, "token_selector_layer", int(cfg.token_selector_layer))
            setattr(model_cfg, "drop_func_name", str(cfg.drop_func_name))

        model_kwargs: Dict[str, Any] = {"config": model_cfg, "torch_dtype": resolved_dtype}
        if cfg.attn_implementation:
            model_kwargs["attn_implementation"] = str(cfg.attn_implementation)
        if device == "cuda":
            model_kwargs["device_map"] = "auto"

        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(cfg.base_model_path, **model_kwargs).eval()
        if device != "cuda":
            self._model.to(device)

        if bool(cfg.use_token_selector):
            self._model.load_token_selector(model_cfg)

        self._processor = Qwen2_5_VLProcessor.from_pretrained(
            cfg.base_model_path,
            min_pixels=int(cfg.min_pixels),
            max_pixels=int(cfg.max_pixels),
        )

    def name(self) -> str:
        return self._cfg.name

    def predict(self, item: QAItem, *, frames_data_urls: Optional[Sequence[str]], prompt: str) -> Prediction:
        import torch
        from PIL import Image

        frames: list[Image.Image] = []
        if frames_data_urls:
            for u in frames_data_urls:
                raw = _decode_data_url_image(u)
                frames.append(Image.open(io.BytesIO(raw)).convert("RGB"))

        system_msg = (
            "You are a strict multiple-choice evaluator. "
            "Reply with only a single letter: A, B, C, or D."
        )
        user_content = []
        if frames:
            # Pass a list of PIL images so FlexSelect can run its token selection on the visual tokens.
            user_content.append({"type": "video", "video": frames})
        user_content.append({"type": "text", "text": prompt})

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]

        # Processor chat template may return a string (single conversation).
        chat_text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=chat_text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        device_dtype = next(self._model.parameters()).dtype
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self._device)
                if torch.is_floating_point(inputs[k]):
                    inputs[k] = inputs[k].to(dtype=device_dtype)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(self._cfg.max_new_tokens),
            "min_new_tokens": 1,
            "do_sample": bool(float(self._cfg.temperature) > 0),
            "temperature": float(self._cfg.temperature),
            "top_p": float(self._cfg.top_p),
            "num_beams": 1,
            "use_cache": True,
        }
        if bool(self._cfg.use_token_selector):
            gen_kwargs["tkn_budget"] = int(self._cfg.tkn_budget)

        with torch.inference_mode():
            out_ids = self._model.generate(**inputs, **gen_kwargs)

        prompt_len = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else 0
        new_ids = out_ids[:, prompt_len:] if prompt_len else out_ids
        text = self._processor.batch_decode(
            new_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        ans = _parse_answer_letter(text)
        return Prediction(
            answer=ans,
            raw=text,
            meta={
                "qa_id": item.id,
                "base_model_path": self._cfg.base_model_path,
                "token_selector_path": self._cfg.token_selector_path,
                "use_token_selector": bool(self._cfg.use_token_selector),
                "token_selector_layer": int(self._cfg.token_selector_layer),
                "drop_func_name": str(self._cfg.drop_func_name),
                "tkn_budget": int(self._cfg.tkn_budget),
                "device": self._device,
                "torch_dtype": self._cfg.torch_dtype,
                "transformers_cache": os.environ.get("TRANSFORMERS_CACHE") or os.environ.get("HF_HOME") or "",
                "n_frames": int(len(frames)),
            },
        )


def build_from_config(cfg: Dict[str, Any]) -> FlexSelectQwen25VLAdapter:
    name = str(cfg.get("name", "FlexSelect-Qwen2.5VL")).strip()
    base_model_path = str(cfg.get("base_model_path", "")).strip()
    token_selector_path = str(cfg.get("token_selector_path", "")).strip()
    if not base_model_path:
        raise ValueError("Missing `base_model_path` in model config.")
    if not token_selector_path:
        raise ValueError("Missing `token_selector_path` in model config.")

    return FlexSelectQwen25VLAdapter(
        FlexSelectQwen25VLConfig(
            name=name,
            base_model_path=base_model_path,
            token_selector_path=token_selector_path,
            use_token_selector=bool(cfg.get("use_token_selector", True)),
            token_selector_layer=int(cfg.get("token_selector_layer", -1)),
            drop_func_name=str(cfg.get("drop_func_name", "token_selection")).strip() or "token_selection",
            tkn_budget=int(cfg.get("tkn_budget", 7040)),
            device=str(cfg.get("device", "auto")).strip() or "auto",
            torch_dtype=str(cfg.get("torch_dtype", "auto")).strip() or "auto",
            attn_implementation=str(cfg.get("attn_implementation", "")).strip() or None,
            max_new_tokens=int(cfg.get("max_new_tokens", 16)),
            temperature=float(cfg.get("temperature", 0.0)),
            top_p=float(cfg.get("top_p", 1.0)),
            min_pixels=int(cfg.get("min_pixels", 3136)),
            max_pixels=int(cfg.get("max_pixels", 16384 * 28 * 28)),
        )
    )
