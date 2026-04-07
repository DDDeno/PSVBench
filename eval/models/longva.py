from __future__ import annotations

import base64
import copy
import io
import os
import re
import sys
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
    except Exception as exc:
        raise ValueError(f"Invalid data URL: {exc}") from exc


def _prepare_longva_imports(longva_root: str) -> None:
    longva_root = os.path.abspath(longva_root)
    longva_pkg_root = longva_root
    if not os.path.isdir(os.path.join(longva_pkg_root, "longva")):
        nested_root = os.path.join(longva_root, "longva")
        if os.path.isdir(os.path.join(nested_root, "longva")):
            longva_pkg_root = nested_root
    if longva_pkg_root not in sys.path:
        sys.path.insert(0, longva_pkg_root)
    longva_mod = sys.modules.get("longva")
    if longva_mod is not None:
        longva_path = getattr(longva_mod, "__file__", "") or ""
        if not longva_path.startswith(longva_root):
            for name in list(sys.modules):
                if name == "longva" or name.startswith("longva."):
                    del sys.modules[name]


def _patch_transformers_modeling_utils() -> None:
    import transformers.modeling_utils as modeling_utils  # type: ignore
    from transformers import pytorch_utils as _pt_utils  # type: ignore

    for _name in (
        "apply_chunking_to_forward",
        "find_pruneable_heads_and_indices",
        "prune_linear_layer",
    ):
        if not hasattr(modeling_utils, _name) and hasattr(_pt_utils, _name):
            setattr(modeling_utils, _name, getattr(_pt_utils, _name))


@dataclass(frozen=True)
class LongVAQwenConfig:
    name: str
    model_path: str
    longva_root: str
    model_base: Optional[str] = None
    model_name: str = "llava_qwen"
    conv_template: str = "qwen_1_5"
    attn_implementation: Optional[str] = None
    device: str = "auto"  # auto|cpu|cuda
    torch_dtype: str = "auto"  # auto|float32|float16|bfloat16
    device_map: str = "auto"
    media: str = "images"  # images|videos
    max_new_tokens: int = 16
    temperature: float = 0.0
    top_p: float = 1.0


class LongVALlavaQwenAdapter(ModelAdapter):
    def __init__(self, cfg: LongVAQwenConfig) -> None:
        self._cfg = cfg
        _prepare_longva_imports(cfg.longva_root)
        _patch_transformers_modeling_utils()

        import torch

        from longva.model.builder import load_pretrained_model  # type: ignore

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        if cfg.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = cfg.device
        self._device = device

        load_kwargs: Dict[str, Any] = {}
        if device == "cuda":
            load_kwargs["device_map"] = str(cfg.device_map)
        else:
            load_kwargs["device_map"] = None

        attn_impl = str(cfg.attn_implementation).strip() if cfg.attn_implementation is not None else ""
        if device != "cuda" and (not attn_impl or attn_impl.startswith("flash_attention")):
            attn_impl = "sdpa"
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl

        model_base = str(cfg.model_base).strip() if cfg.model_base else ""
        if not model_base:
            weight_names = [
                "pytorch_model.bin",
                "pytorch_model.bin.index.json",
                "model.safetensors",
                "model.safetensors.index.json",
            ]
            has_root_weights = any(
                os.path.isfile(os.path.join(cfg.model_path, name)) for name in weight_names
            )
            llm_path = os.path.join(cfg.model_path, "llm")
            if not has_root_weights and os.path.isdir(llm_path):
                has_llm_weights = any(
                    os.path.isfile(os.path.join(llm_path, name)) for name in weight_names
                )
                if has_llm_weights:
                    model_base = llm_path

        tokenizer, model, image_processor, _max_length = load_pretrained_model(
            cfg.model_path,
            model_base or None,
            str(cfg.model_name),
            **{k: v for k, v in load_kwargs.items() if v is not None},
        )
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        self._image_processor = image_processor

    def name(self) -> str:
        return self._cfg.name

    def _tokenize_prompt(self, prompt: str) -> Any:
        from longva.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN  # type: ignore
        from longva.conversation import conv_templates  # type: ignore
        from longva.mm_utils import tokenizer_image_token  # type: ignore

        system_msg = (
            "You are a strict multiple-choice evaluator. "
            "Reply with only a single letter: A, B, C, or D."
        )
        user_text = DEFAULT_IMAGE_TOKEN + "\n" + system_msg + "\n\n" + prompt

        conv = copy.deepcopy(conv_templates[str(self._cfg.conv_template)])
        conv.append_message(conv.roles[0], user_text)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()
        return tokenizer_image_token(
            prompt_text,
            self._tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0)

    def predict(self, item: QAItem, *, frames_data_urls: Optional[Sequence[str]], prompt: str) -> Prediction:
        import numpy as np
        import torch
        from PIL import Image

        images: list[Image.Image] = []
        if frames_data_urls:
            for u in frames_data_urls:
                raw, _mime = _decode_data_url_image(u)
                images.append(Image.open(io.BytesIO(raw)).convert("RGB"))

        input_ids = self._tokenize_prompt(prompt).to(self._device)
        attention_mask = torch.ones_like(input_ids)

        gen_kwargs: Dict[str, Any] = {"max_new_tokens": int(self._cfg.max_new_tokens)}
        if float(self._cfg.temperature) > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(self._cfg.temperature)
            gen_kwargs["top_p"] = float(self._cfg.top_p)
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            if images and self._cfg.media == "videos":
                clip = np.stack([np.asarray(im) for im in images], axis=0)  # (T,H,W,3)
                pixel_values = self._image_processor.preprocess(clip, return_tensors="pt")["pixel_values"]
                pixel_values = pixel_values.to(device=self._device)
                try:
                    pixel_values = pixel_values.to(dtype=next(self._model.parameters()).dtype)
                except Exception:
                    pass
                videos = [pixel_values]
                out_ids = self._model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    images=videos,
                    image_sizes=[img.size for img in images],
                    modalities=["video"],
                    **gen_kwargs,
                )
            elif images:
                from longva.mm_utils import process_images  # type: ignore

                image_tensor = process_images(images, self._image_processor, self._model.config)
                if isinstance(image_tensor, torch.Tensor):
                    image_tensor = image_tensor.to(device=self._device)
                    try:
                        image_tensor = image_tensor.to(dtype=next(self._model.parameters()).dtype)
                    except Exception:
                        pass
                    images_arg = image_tensor
                else:
                    images_arg = image_tensor
                out_ids = self._model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    images=images_arg,
                    image_sizes=[img.size for img in images],
                    modalities=["image"],
                    **gen_kwargs,
                )
            else:
                out_ids = self._model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )

        text = self._tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
        ans = _parse_answer_letter(text)
        return Prediction(
            answer=ans,
            raw=text,
            meta={
                "qa_id": item.id,
                "model_path": self._cfg.model_path,
                "longva_root": self._cfg.longva_root,
                "device": self._device,
                "torch_dtype": self._cfg.torch_dtype,
                "transformers_cache": os.environ.get("TRANSFORMERS_CACHE") or os.environ.get("HF_HOME") or "",
            },
        )


def build_from_config(cfg: dict) -> LongVALlavaQwenAdapter:
    name = str(cfg.get("name", "LongVA")).strip()
    model_path = str(cfg.get("model_path", "")).strip()
    longva_root = str(cfg.get("longva_root", "")).strip()
    if not model_path:
        raise ValueError("Missing `model_path` in model config.")
    if not longva_root:
        raise ValueError("Missing `longva_root` in model config.")
    media = str(cfg.get("media", "images")).strip().lower() or "images"
    if media not in {"images", "videos"}:
        raise ValueError(f"Unsupported media: {media} (expected images|videos)")
    return LongVALlavaQwenAdapter(
        LongVAQwenConfig(
            name=name,
            model_path=model_path,
            longva_root=longva_root,
            model_base=str(cfg.get("model_base", "")).strip() or None,
            model_name=str(cfg.get("model_name", "llava_qwen")).strip() or "llava_qwen",
            conv_template=str(cfg.get("conv_template", "qwen_1_5")).strip() or "qwen_1_5",
            attn_implementation=str(cfg.get("attn_implementation", "")).strip() or None,
            device=str(cfg.get("device", "auto")).strip() or "auto",
            torch_dtype=str(cfg.get("torch_dtype", "auto")).strip() or "auto",
            device_map=str(cfg.get("device_map", "auto")).strip() or "auto",
            media=media,
            max_new_tokens=int(cfg.get("max_new_tokens", 16)),
            temperature=float(cfg.get("temperature", 0.0)),
            top_p=float(cfg.get("top_p", 1.0)),
        )
    )
