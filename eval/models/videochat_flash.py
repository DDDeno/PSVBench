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


def _load_py_module(module_name: str, path: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tokenizer_image_token(prompt: str, tokenizer: Any, *, image_token: str, image_token_index: int):
    # Minimal re-implementation of VideoChat-Flash's `tokenizer_image_token` to avoid
    # importing `mm_utils.py` (which pulls in optional video deps).
    parts = prompt.split(str(image_token))
    input_ids = []
    for i, part in enumerate(parts):
        ids = tokenizer.encode(part, add_special_tokens=(i == 0))
        input_ids.extend(ids)
        if i != len(parts) - 1:
            input_ids.append(int(image_token_index))
    return input_ids


def _ensure_transformers_generate() -> None:
    """
    transformers>=4.50: `PreTrainedModel` no longer inherits `GenerationMixin`.

    Some trust_remote_code model implementations (including VideoChat-Flash) still call
    `super().generate(...)`, which breaks with:
      AttributeError: 'super' object has no attribute 'generate'

    Work around by copying missing generation methods from `GenerationMixin` onto
    `PreTrainedModel`, restoring the pre-4.50 behavior.
    """
    try:
        from transformers.generation.utils import GenerationMixin  # type: ignore
        from transformers.modeling_utils import PreTrainedModel  # type: ignore
    except Exception:
        return

    for name in dir(GenerationMixin):
        if name.startswith("__"):
            continue
        attr = getattr(GenerationMixin, name, None)
        if attr is None or not callable(attr):
            continue
        if not hasattr(PreTrainedModel, name):
            try:
                raw_attr = GenerationMixin.__dict__.get(name)
                if isinstance(raw_attr, staticmethod):
                    setattr(PreTrainedModel, name, staticmethod(raw_attr.__func__))
                elif isinstance(raw_attr, classmethod):
                    setattr(PreTrainedModel, name, classmethod(raw_attr.__func__))
                else:
                    setattr(PreTrainedModel, name, attr)
            except Exception:
                # Best-effort: some attributes might be read-only in some envs.
                pass


def _ensure_generation_config(model: Any) -> None:
    """
    Some trust_remote_code models ship with `generation_config=None`, which breaks
    transformers' `GenerationMixin.generate()` in newer versions.
    """
    try:
        from transformers import GenerationConfig  # type: ignore
    except Exception:
        return

    if getattr(model, "generation_config", None) is not None:
        return
    try:
        model.generation_config = GenerationConfig.from_model_config(model.config)
    except Exception:
        try:
            model.generation_config = GenerationConfig()
        except Exception:
            pass


@dataclass(frozen=True)
class VideoChatFlashConfig:
    name: str
    model_path: str
    media: str = "videos"  # images|videos
    device: str = "auto"  # auto|cpu|cuda
    torch_dtype: str = "auto"  # auto|float32|float16|bfloat16
    attn_implementation: Optional[str] = None
    max_new_tokens: int = 16
    temperature: float = 0.0
    top_p: float = 1.0


class VideoChatFlashAdapter(ModelAdapter):
    def __init__(self, cfg: VideoChatFlashConfig) -> None:
        self._cfg = cfg

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        _ensure_transformers_generate()

        if cfg.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = cfg.device

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

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if resolved_torch_dtype is not None:
            model_kwargs["torch_dtype"] = resolved_torch_dtype
        if cfg.attn_implementation and not (
            device != "cuda" and str(cfg.attn_implementation).startswith("flash_attention")
        ):
            model_kwargs["attn_implementation"] = str(cfg.attn_implementation)
        if device == "cuda":
            model_kwargs["device_map"] = "auto"

        self._tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(cfg.model_path, **model_kwargs)
        _ensure_generation_config(self._model)
        if device != "cuda":
            self._model.to(device)
        self._model.eval()
        self._device = device

        constants = _load_py_module(
            "videochat_flash_constants",
            os.path.join(cfg.model_path, "constants.py"),
        )
        conversation = _load_py_module(
            "videochat_flash_conversation",
            os.path.join(cfg.model_path, "conversation.py"),
        )
        self._image_token = str(getattr(constants, "DEFAULT_IMAGE_TOKEN", "<image>"))
        self._image_token_index = int(getattr(constants, "IMAGE_TOKEN_INDEX", -200))
        self._conv_templates = getattr(conversation, "conv_templates")

        vision_tower = self._model.get_vision_tower()
        self._image_processor = getattr(vision_tower, "image_processor", None)
        if self._image_processor is None:
            raise RuntimeError("VideoChat-Flash vision tower missing `image_processor`.")

    def name(self) -> str:
        return self._cfg.name

    def predict(self, item: QAItem, *, frames_data_urls: Optional[Sequence[str]], prompt: str) -> Prediction:
        import numpy as np
        import torch
        from PIL import Image

        frames = []
        if frames_data_urls:
            for u in frames_data_urls:
                raw, _mime = _decode_data_url_image(u)
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                frames.append(np.asarray(img))

        system_msg = (
            "You are a strict multiple-choice evaluator. "
            "Reply with only a single letter: A, B, C, or D."
        )

        conv = self._conv_templates["qwen_2"].copy()
        if frames:
            user_text = f"{self._image_token}\n{system_msg}\n\n{prompt}"
        else:
            user_text = f"{system_msg}\n\n{prompt}"
        conv.append_message(conv.roles[0], user_text)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        input_ids = _tokenizer_image_token(
            prompt_text,
            self._tokenizer,
            image_token=self._image_token,
            image_token_index=self._image_token_index,
        )
        input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=self._device).unsqueeze(0)

        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.bos_token_id
        attention_mask = input_ids_t.ne(int(self._tokenizer.pad_token_id)).long()

        gen_kwargs: Dict[str, Any] = {"max_new_tokens": int(self._cfg.max_new_tokens)}
        if float(self._cfg.temperature) > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(self._cfg.temperature)
            gen_kwargs["top_p"] = float(self._cfg.top_p)
        else:
            gen_kwargs["do_sample"] = False

        with torch.inference_mode():
            # Some VideoChat-Flash checkpoints set `generation_config=None`.
            _ensure_generation_config(self._model)
            if frames and self._cfg.media == "videos":
                image_sizes = [frames[0].shape[:2]]
                pixel_values = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
                pixel_values = pixel_values.to(device=self._device)
                try:
                    pixel_values = pixel_values.to(dtype=next(self._model.parameters()).dtype)
                except Exception:
                    pass
                out_ids = self._model.generate(
                    inputs=input_ids_t,
                    images=[pixel_values],
                    attention_mask=attention_mask,
                    modalities=["video"],
                    image_sizes=image_sizes,
                    use_cache=True,
                    **gen_kwargs,
                )
            elif frames:
                # Fallback: treat as multi-image inputs by passing frames as a list.
                out_ids = self._model.generate(
                    inputs=input_ids_t,
                    images=frames,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )
            else:
                out_ids = self._model.generate(inputs=input_ids_t, attention_mask=attention_mask, **gen_kwargs)

        text = self._tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
        ans = _parse_answer_letter(text)
        return Prediction(
            answer=ans,
            raw=text,
            meta={
                "qa_id": item.id,
                "model_path": self._cfg.model_path,
                "device": self._device,
                "torch_dtype": self._cfg.torch_dtype,
                "transformers_cache": os.environ.get("TRANSFORMERS_CACHE") or os.environ.get("HF_HOME") or "",
            },
        )


def build_from_config(cfg: dict) -> VideoChatFlashAdapter:
    name = str(cfg.get("name", "videochat_flash")).strip()
    model_path = str(cfg.get("model_path", "")).strip()
    if not model_path:
        raise ValueError("Missing `model_path` in model config.")
    media = str(cfg.get("media", "videos")).strip().lower() or "videos"
    if media not in {"images", "videos"}:
        raise ValueError(f"Unsupported media: {media} (expected images|videos)")
    return VideoChatFlashAdapter(
        VideoChatFlashConfig(
            name=name,
            model_path=model_path,
            media=media,
            device=str(cfg.get("device", "auto")).strip() or "auto",
            torch_dtype=str(cfg.get("torch_dtype", "auto")).strip() or "auto",
            attn_implementation=str(cfg.get("attn_implementation", "")).strip() or None,
            max_new_tokens=int(cfg.get("max_new_tokens", 16)),
            temperature=float(cfg.get("temperature", 0.0)),
            top_p=float(cfg.get("top_p", 1.0)),
        )
    )
