"""
Generic HuggingFace transformers model adapter.

Supports Qwen-VL, LLaVA-OneVision, Aria, VideoLLaMA3, and other
AutoModelForVision2Seq / AutoModelForCausalLM checkpoints from HF Hub.

For InternVL3/3.5, use the dedicated ``internvl_chat`` adapter.
For VILA/nVILA, use the dedicated ``vila`` adapter.
"""
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
    return m.group(1).upper() if m else None


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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HFTransformersConfig:
    name: str
    model_path: str
    tokenizer_path: Optional[str] = None
    processor_path: Optional[str] = None
    device: str = "auto"
    torch_dtype: str = "auto"
    media: str = "images"
    trust_remote_code: bool = False
    attn_implementation: Optional[str] = None
    max_new_tokens: int = 16
    temperature: float = 0.0
    top_p: float = 1.0
    min_pixels: Optional[int] = None
    max_pixels: Optional[int] = None


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class HFTransformersAdapter(ModelAdapter):

    def __init__(self, cfg: HFTransformersConfig) -> None:
        self._cfg = cfg

        import torch
        from transformers import AutoConfig

        device = "cuda" if (cfg.device == "auto" and torch.cuda.is_available()) else cfg.device
        if device == "auto":
            device = "cpu"
        self._device = device

        resolved_torch_dtype = {
            "auto": None, "float32": torch.float32,
            "float16": torch.float16, "bfloat16": torch.bfloat16,
        }.get(cfg.torch_dtype)
        if cfg.torch_dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError(f"Unsupported torch_dtype: {cfg.torch_dtype}")

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": bool(cfg.trust_remote_code),
            "low_cpu_mem_usage": True,
        }
        if resolved_torch_dtype is not None:
            model_kwargs["torch_dtype"] = resolved_torch_dtype
        if cfg.attn_implementation:
            model_kwargs["attn_implementation"] = str(cfg.attn_implementation)
        if device == "cuda":
            model_kwargs["device_map"] = "auto"

        resolved_config = AutoConfig.from_pretrained(cfg.model_path, trust_remote_code=bool(cfg.trust_remote_code))
        self._model_type = str(getattr(resolved_config, "model_type", "")).strip()
        architectures = getattr(resolved_config, "architectures", None) or []

        self._use_aria = (
            self._model_type == "aria"
            or "AriaForConditionalGeneration" in set(map(str, architectures))
        )
        self._use_videollama3 = self._model_type.startswith("videollama3")

        # Load model + processor
        if self._use_aria:
            self._init_aria(cfg, model_kwargs, device)
        elif self._use_videollama3:
            self._init_videollama3(cfg, model_kwargs, device)
        else:
            self._init_generic(cfg, model_kwargs, device)

        # LLaVA-OneVision needs left padding
        if self._model_type == "llava_onevision" and self._processor is not None:
            tok = getattr(self._processor, "tokenizer", None)
            if tok is not None:
                tok.padding_side = "left"

    def _init_aria(self, cfg, model_kwargs, device):
        from transformers import AriaForConditionalGeneration, AriaProcessor

        self._processor = AriaProcessor.from_pretrained(cfg.model_path, trust_remote_code=True)
        self._tokenizer = getattr(self._processor, "tokenizer", None)
        self._model = AriaForConditionalGeneration.from_pretrained(cfg.model_path, **model_kwargs)
        if device != "cuda":
            self._model.to(device)
        self._model.eval()

    def _init_videollama3(self, cfg, model_kwargs, device):
        from transformers import AutoModelForCausalLM, AutoProcessor

        processor_src = str(cfg.processor_path or cfg.model_path)
        proc_kwargs: Dict[str, Any] = {"trust_remote_code": bool(cfg.trust_remote_code)}
        if cfg.min_pixels is not None:
            proc_kwargs["min_pixels"] = int(cfg.min_pixels)
        if cfg.max_pixels is not None:
            proc_kwargs["max_pixels"] = int(cfg.max_pixels)

        try:
            self._processor = AutoProcessor.from_pretrained(processor_src, **proc_kwargs)
        except Exception:
            self._processor = None

        self._model = AutoModelForCausalLM.from_pretrained(cfg.model_path, **model_kwargs)
        if device != "cuda":
            self._model.to(device)
        self._model.eval()

        if self._processor is not None:
            self._tokenizer = getattr(self._processor, "tokenizer", None)
        else:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(cfg.tokenizer_path or cfg.model_path), trust_remote_code=True)

    def _init_generic(self, cfg, model_kwargs, device):
        from transformers import AutoProcessor

        processor_src = str(cfg.processor_path or cfg.model_path)
        proc_kwargs: Dict[str, Any] = {"trust_remote_code": bool(cfg.trust_remote_code)}
        if cfg.min_pixels is not None:
            proc_kwargs["min_pixels"] = int(cfg.min_pixels)
        if cfg.max_pixels is not None:
            proc_kwargs["max_pixels"] = int(cfg.max_pixels)

        try:
            self._processor = AutoProcessor.from_pretrained(processor_src, **proc_kwargs)
        except TypeError:
            # Some processors don't accept min/max_pixels
            proc_kwargs.pop("min_pixels", None)
            proc_kwargs.pop("max_pixels", None)
            self._processor = AutoProcessor.from_pretrained(processor_src, **proc_kwargs)

        self._tokenizer = None

        try:
            from transformers import AutoModelForVision2Seq
            self._model = AutoModelForVision2Seq.from_pretrained(cfg.model_path, **model_kwargs)
        except Exception:
            from transformers import AutoModelForCausalLM
            self._model = AutoModelForCausalLM.from_pretrained(cfg.model_path, **model_kwargs)

        if device != "cuda":
            self._model.to(device)
        self._model.eval()

    def name(self) -> str:
        return self._cfg.name

    # -- Decode helper -------------------------------------------------------

    def _decode_generated(self, inputs: Dict[str, Any], out_ids) -> str:
        import torch

        prompt_len = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else 0
        new_ids = out_ids[:, prompt_len:] if prompt_len else out_ids
        if hasattr(new_ids, "shape") and int(new_ids.shape[1]) == 0:
            new_ids = out_ids

        if hasattr(self._processor, "batch_decode"):
            text = self._processor.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            if text.strip():
                return text
            return self._processor.batch_decode(new_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

        if self._tokenizer is not None:
            return self._tokenizer.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        raise RuntimeError("No decoder available.")

    # -- Generation kwargs ---------------------------------------------------

    def _gen_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"max_new_tokens": int(self._cfg.max_new_tokens), "min_new_tokens": 1}
        if float(self._cfg.temperature) > 0:
            kwargs["do_sample"] = True
            kwargs["temperature"] = float(self._cfg.temperature)
            kwargs["top_p"] = float(self._cfg.top_p)
        else:
            kwargs["do_sample"] = False
        return kwargs

    # -- VideoLLaMA3 path ----------------------------------------------------

    def _predict_videollama3(self, images, prompt: str) -> str:
        import torch

        system_msg = "You are a strict multiple-choice evaluator. Reply with only a single letter: A, B, C, or D."
        user_content = []
        if images:
            user_content.append({"type": "video", "video": images, "num_frames": len(images)})
        user_content.append({"type": "text", "text": prompt})

        conversation = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]

        with torch.inference_mode():
            inputs = self._processor(conversation=conversation, return_tensors="pt", add_generation_prompt=True)
            device_dtype = next(self._model.parameters()).dtype
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self._device)
                    if torch.is_floating_point(inputs[k]):
                        inputs[k] = inputs[k].to(dtype=device_dtype)

            gen_kwargs = self._gen_kwargs()
            out_ids = self._model.generate(**inputs, **gen_kwargs)
            return self._decode_generated(inputs, out_ids)

    # -- Generic multimodal path ---------------------------------------------

    def _predict_generic(self, images, prompt: str) -> str:
        import numpy as np
        import torch

        system_msg = "You are a strict multiple-choice evaluator. Reply with only a single letter: A, B, C, or D."

        proc_kwargs: Dict[str, Any] = {"return_tensors": "pt", "padding": True}

        if images and self._cfg.media == "videos" and self._model_type in {"llava_next_video", "llava_onevision"}:
            clip = np.stack([np.asarray(im) for im in images], axis=0)
            if self._model_type == "llava_next_video":
                content = [{"type": "text", "text": prompt}, {"type": "video"}]
            else:
                content = [{"type": "video"}, {"type": "text", "text": prompt}]
            messages = [{"role": "user", "content": content}]

            if hasattr(self._processor, "apply_chat_template"):
                try:
                    chat_text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                    chat_text = "USER: <video>\n" + system_msg + "\n\n" + prompt + "\nASSISTANT:"
            else:
                chat_text = "USER: <video>\n" + system_msg + "\n\n" + prompt + "\nASSISTANT:"
            inputs = self._processor(text=[chat_text], videos=[clip], **proc_kwargs)

        elif images and self._cfg.media == "videos":
            clip = np.stack([np.asarray(im) for im in images], axis=0)
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]},
            ]
            if hasattr(self._processor, "apply_chat_template"):
                try:
                    chat_text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                    chat_text = "USER: <video>\n" + system_msg + "\n\n" + prompt + "\nASSISTANT:"
            else:
                chat_text = "USER: <video>\n" + system_msg + "\n\n" + prompt + "\nASSISTANT:"
            inputs = self._processor(text=[chat_text], videos=[clip], **proc_kwargs)

        elif images:
            user_content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": prompt}]
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_content},
            ]
            if hasattr(self._processor, "apply_chat_template"):
                try:
                    chat_text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                    chat_text = system_msg + "\n\n" + prompt
            else:
                chat_text = system_msg + "\n\n" + prompt
            inputs = self._processor(text=[chat_text], images=images, **proc_kwargs)
        else:
            chat_text = system_msg + "\n\n" + prompt
            inputs = self._processor(text=[chat_text], **proc_kwargs)

        device_dtype = next(self._model.parameters()).dtype
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                v = v.to(self._device)
                if torch.is_floating_point(v) and v.dtype != device_dtype:
                    v = v.to(dtype=device_dtype)
                inputs[k] = v

        with torch.inference_mode():
            out_ids = self._model.generate(**inputs, **self._gen_kwargs())

        return self._decode_generated(inputs, out_ids)

    # -- Main predict --------------------------------------------------------

    def predict(self, item: QAItem, *, frames_data_urls: Optional[Sequence[str]], prompt: str) -> Prediction:
        from PIL import Image

        images = []
        if frames_data_urls:
            for u in frames_data_urls:
                raw, _ = _decode_data_url_image(u)
                images.append(Image.open(io.BytesIO(raw)).convert("RGB"))

        if self._use_videollama3 and self._processor is not None:
            text = self._predict_videollama3(images, prompt)
        else:
            text = self._predict_generic(images, prompt)

        ans = _parse_answer_letter(text)
        return Prediction(answer=ans, raw=text, meta={"model_path": self._cfg.model_path, "device": self._device})


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_from_config(cfg: dict) -> HFTransformersAdapter:
    name = str(cfg.get("name", "hf_transformers")).strip()
    model_path = str(cfg.get("model_path", "")).strip()
    if not model_path:
        raise ValueError("Missing `model_path` in model config.")
    media = str(cfg.get("media", "images")).strip().lower() or "images"
    if media not in {"images", "videos"}:
        raise ValueError(f"Unsupported media: {media}")
    return HFTransformersAdapter(
        HFTransformersConfig(
            name=name,
            model_path=model_path,
            tokenizer_path=str(cfg.get("tokenizer_path", "")).strip() or None,
            processor_path=str(cfg.get("processor_path", "")).strip() or None,
            device=str(cfg.get("device", "auto")).strip() or "auto",
            torch_dtype=str(cfg.get("torch_dtype", "auto")).strip() or "auto",
            media=media,
            trust_remote_code=bool(cfg.get("trust_remote_code", False)),
            attn_implementation=str(cfg.get("attn_implementation", "")).strip() or None,
            max_new_tokens=int(cfg.get("max_new_tokens", 16)),
            temperature=float(cfg.get("temperature", 0.0)),
            top_p=float(cfg.get("top_p", 1.0)),
            min_pixels=int(cfg["min_pixels"]) if cfg.get("min_pixels") is not None else None,
            max_pixels=int(cfg["max_pixels"]) if cfg.get("max_pixels") is not None else None,
        )
    )
