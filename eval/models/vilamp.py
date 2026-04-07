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


def _prepare_vilamp_imports(vilamp_root: str) -> None:
    vilamp_root = os.path.abspath(vilamp_root)
    if vilamp_root not in sys.path:
        sys.path.insert(0, vilamp_root)
    llava_mod = sys.modules.get("llava")
    if llava_mod is not None:
        llava_path = getattr(llava_mod, "__file__", "") or ""
        if not llava_path.startswith(vilamp_root):
            for name in list(sys.modules):
                if name == "llava" or name.startswith("llava."):
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


def _ensure_sentence_transformers() -> None:
    try:
        from sentence_transformers import util as _util  # noqa: F401
        return
    except Exception:
        pass

    import types
    import torch
    import torch.nn.functional as F

    def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.dim() == 1:
            a = a.unsqueeze(0)
        if b.dim() == 1:
            b = b.unsqueeze(0)
        a = F.normalize(a, p=2, dim=-1)
        b = F.normalize(b, p=2, dim=-1)
        return a @ b.T

    util_mod = types.ModuleType("sentence_transformers.util")
    util_mod.cos_sim = _cos_sim  # type: ignore[attr-defined]

    st_mod = types.ModuleType("sentence_transformers")
    st_mod.util = util_mod  # type: ignore[attr-defined]

    sys.modules.setdefault("sentence_transformers", st_mod)
    sys.modules.setdefault("sentence_transformers.util", util_mod)


@dataclass(frozen=True)
class ViLAMPQwenConfig:
    name: str
    model_path: str
    vilamp_root: str
    clip_model_path: Optional[str] = None
    model_name: str = "llava_qwen"
    conv_template: str = "qwen_1_5"
    attn_implementation: Optional[str] = None
    device: str = "auto"  # auto|cpu|cuda
    torch_dtype: str = "auto"  # auto|float32|float16|bfloat16
    device_map: str = "auto"
    media: str = "videos"  # images|videos
    max_new_tokens: int = 16
    temperature: float = 0.0
    top_p: float = 1.0


class ViLAMPLlavaQwenAdapter(ModelAdapter):
    def __init__(self, cfg: ViLAMPQwenConfig) -> None:
        self._cfg = cfg
        _prepare_vilamp_imports(cfg.vilamp_root)
        _patch_transformers_modeling_utils()
        _ensure_sentence_transformers()

        import torch
        from transformers import CLIPProcessor

        from llava.model.builder import load_pretrained_model  # type: ignore

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        if cfg.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = cfg.device
        if device != "cuda":
            raise RuntimeError("ViLAMP-llava-qwen_local is pinned to CUDA; set device=cuda.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False. Check driver/CUDA_VISIBLE_DEVICES.")
        gpu_index = 0
        if cfg.device_map:
            dm = str(cfg.device_map).strip()
            if dm.startswith("cuda:"):
                try:
                    gpu_index = int(dm.split(":", 1)[1])
                except ValueError:
                    pass
        torch.cuda.set_device(gpu_index)
        self._device = device

        load_kwargs: Dict[str, Any] = {}
        attn_impl = str(cfg.attn_implementation).strip() if cfg.attn_implementation is not None else ""
        if not attn_impl:
            attn_impl = "flash_attention_2"
        if not attn_impl.startswith("flash_attention"):
            raise RuntimeError("ViLAMP-llava-qwen_local is pinned to flash_attention_2.")
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl

        overwrite_config: Optional[Dict[str, Any]] = None
        if cfg.clip_model_path:
            overwrite_config = {"clip_model_path": str(cfg.clip_model_path)}

        tokenizer, model, image_processor, _max_length = load_pretrained_model(
            cfg.model_path,
            None,
            str(cfg.model_name),
            overwrite_config=overwrite_config,
            **{k: v for k, v in load_kwargs.items() if v is not None},
        )
        model.eval()
        self._model_device = torch.device(f"cuda:{gpu_index}")
        model.to(self._model_device)
        if any(p.device.type != "cuda" for p in model.parameters()):
            raise RuntimeError("Model parameters are not on CUDA.")
        frame_selector = None
        if hasattr(model, "get_frame_selector"):
            frame_selector = model.get_frame_selector()
        elif hasattr(model, "get_model") and hasattr(model.get_model(), "get_frame_selector"):
            frame_selector = model.get_model().get_frame_selector()
        if frame_selector is not None:
            frame_selector.to(self._model_device)

        clip_model_path = cfg.clip_model_path or getattr(model.config, "clip_model_path", "clip-ViT-B-32/0_CLIPModel")
        self._clip_processor = CLIPProcessor.from_pretrained(clip_model_path, local_files_only=True)
        self._tokenizer = tokenizer
        self._model = model
        self._image_processor = image_processor

    def name(self) -> str:
        return self._cfg.name

    def _tokenize_prompt(self, prompt: str) -> Any:
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN  # type: ignore
        from llava.conversation import conv_templates  # type: ignore
        from llava.mm_utils import tokenizer_image_token  # type: ignore

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

    def _build_selector_inputs(self, frames: Sequence["Image.Image"], prompt: str) -> Dict[str, Any]:
        from transformers import CLIPTextConfig

        image_inputs = self._clip_processor(images=list(frames), return_tensors="pt").to(self._model_device)
        query_inputs = self._clip_processor(
            text=[prompt],
            return_tensors="pt",
            truncation=True,
            max_length=CLIPTextConfig().max_position_embeddings,
        ).to(self._model_device)
        kwargs: Dict[str, Any] = {"image_inputs": [image_inputs], "query_inputs": [query_inputs]}
        kwargs["offload_params"] = list(kwargs.keys())
        return kwargs

    def predict(self, item: QAItem, *, frames_data_urls: Optional[Sequence[str]], prompt: str) -> Prediction:
        import numpy as np
        import torch
        from PIL import Image

        frames: list[Image.Image] = []
        if frames_data_urls:
            for u in frames_data_urls:
                raw, _mime = _decode_data_url_image(u)
                frames.append(Image.open(io.BytesIO(raw)).convert("RGB"))

        input_ids = self._tokenize_prompt(prompt).to(self._model_device)
        attention_mask = torch.ones_like(input_ids)

        gen_kwargs: Dict[str, Any] = {"max_new_tokens": int(self._cfg.max_new_tokens)}
        if float(self._cfg.temperature) > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(self._cfg.temperature)
            gen_kwargs["top_p"] = float(self._cfg.top_p)
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            if frames and self._cfg.media == "videos":
                clip = np.stack([np.asarray(im) for im in frames], axis=0)  # (T,H,W,3)
                pixel_values = self._image_processor.preprocess(clip, return_tensors="pt")["pixel_values"]
                pixel_values = pixel_values.to(device=self._model_device)
                try:
                    pixel_values = pixel_values.to(dtype=next(self._model.parameters()).dtype)
                except Exception:
                    pass
                videos = [pixel_values]
                selector_kwargs = self._build_selector_inputs(frames, prompt)
                out_ids = self._model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    images=videos,
                    image_sizes=[img.size for img in frames],
                    modalities=["video"],
                    **gen_kwargs,
                    **selector_kwargs,
                )
            elif frames:
                from llava.mm_utils import process_images  # type: ignore

                image_tensor = process_images(frames, self._image_processor, self._model.config)
                if isinstance(image_tensor, torch.Tensor):
                    image_tensor = image_tensor.to(device=self._model_device)
                    try:
                        image_tensor = image_tensor.to(dtype=next(self._model.parameters()).dtype)
                    except Exception:
                        pass
                    images_arg = image_tensor
                else:
                    images_arg = image_tensor
                selector_kwargs = self._build_selector_inputs(frames, prompt)
                out_ids = self._model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    images=images_arg,
                    image_sizes=[img.size for img in frames],
                    modalities=["image"],
                    **gen_kwargs,
                    **selector_kwargs,
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
                "vilamp_root": self._cfg.vilamp_root,
                "device": self._device,
                "torch_dtype": self._cfg.torch_dtype,
                "transformers_cache": os.environ.get("TRANSFORMERS_CACHE") or os.environ.get("HF_HOME") or "",
            },
        )


def build_from_config(cfg: dict) -> ViLAMPLlavaQwenAdapter:
    name = str(cfg.get("name", "ViLAMP")).strip()
    model_path = str(cfg.get("model_path", "")).strip()
    vilamp_root = str(cfg.get("vilamp_root", "")).strip()
    if not model_path:
        raise ValueError("Missing `model_path` in model config.")
    if not vilamp_root:
        raise ValueError("Missing `vilamp_root` in model config.")
    media = str(cfg.get("media", "videos")).strip().lower() or "videos"
    if media not in {"images", "videos"}:
        raise ValueError(f"Unsupported media: {media} (expected images|videos)")
    clip_model_path = str(cfg.get("clip_model_path", "")).strip() or None
    return ViLAMPLlavaQwenAdapter(
        ViLAMPQwenConfig(
            name=name,
            model_path=model_path,
            vilamp_root=vilamp_root,
            clip_model_path=clip_model_path,
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
