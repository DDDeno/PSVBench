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


def _prepare_vila_imports(vila_root: str) -> None:
    vila_root = os.path.abspath(vila_root)
    if vila_root not in sys.path:
        sys.path.insert(0, vila_root)
    llava_mod = sys.modules.get("llava")
    if llava_mod is not None:
        llava_path = getattr(llava_mod, "__file__", "") or ""
        if not llava_path.startswith(vila_root):
            for name in list(sys.modules):
                if name == "llava" or name.startswith("llava."):
                    del sys.modules[name]


def _ensure_s2wrapper() -> None:
    if "s2wrapper" in sys.modules:
        return
    # s2wrapper must be installed: pip install s2wrapper
    # Or set S2WRAPPER_ROOT env var to the cloned repo path.


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


def _configure_ps3_and_context_length(model: Any) -> None:
    num_look_close = os.environ.get("NUM_LOOK_CLOSE", None)
    num_token_look_close = os.environ.get("NUM_TOKEN_LOOK_CLOSE", None)
    select_num_each_scale = os.environ.get("SELECT_NUM_EACH_SCALE", None)
    look_close_mode = os.environ.get("LOOK_CLOSE_MODE", None)
    smooth_selection_prob = os.environ.get("SMOOTH_SELECTION_PROB", None)

    if num_look_close is not None:
        model.num_look_close = int(num_look_close)
    if num_token_look_close is not None:
        model.num_token_look_close = int(num_token_look_close)
    if select_num_each_scale is not None:
        select_num_each_scale = [int(x) for x in select_num_each_scale.split("+")]
        model.get_vision_tower().vision_tower.vision_model.max_select_num_each_scale = select_num_each_scale
    if look_close_mode is not None:
        model.look_close_mode = look_close_mode
    if smooth_selection_prob is not None:
        if smooth_selection_prob.lower() == "true":
            model.smooth_selection_prob = True
        elif smooth_selection_prob.lower() == "false":
            model.smooth_selection_prob = False
        else:
            raise ValueError(f"Invalid smooth selection prob: {smooth_selection_prob}")

    context_length = model.tokenizer.model_max_length
    if num_look_close is not None:
        context_length = max(context_length, int(num_look_close) * 2560 // 4 + 1024)
    if num_token_look_close is not None:
        context_length = max(context_length, int(num_token_look_close) // 4 + 1024)
    context_length = max(getattr(model.tokenizer, "model_max_length", context_length), context_length)
    model.config.model_max_length = context_length
    model.config.tokenizer_model_max_length = context_length
    model.llm.config.model_max_length = context_length
    model.llm.config.tokenizer_model_max_length = context_length
    model.tokenizer.model_max_length = context_length


@dataclass(frozen=True)
class VILANVILAConfig:
    name: str
    model_path: str
    vila_root: str
    model_name: str = "nvila-8b"
    conv_template: str = "vicuna_v1"
    attn_implementation: Optional[str] = None
    device: str = "auto"  # auto|cpu|cuda
    torch_dtype: str = "auto"  # auto|float32|float16|bfloat16
    device_map: str = "auto"
    media: str = "videos"  # images|videos
    max_new_tokens: int = 16
    temperature: float = 0.0
    top_p: float = 1.0


class VILANVILAAdapter(ModelAdapter):
    def __init__(self, cfg: VILANVILAConfig) -> None:
        self._cfg = cfg
        _prepare_vila_imports(cfg.vila_root)
        _ensure_s2wrapper()
        _patch_transformers_modeling_utils()

        import torch

        import llava  # type: ignore
        from llava import conversation as clib  # type: ignore

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        if cfg.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = cfg.device
        if device != "cuda":
            raise RuntimeError("VILA NVILA adapter requires CUDA; set device=cuda.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False. Check driver/CUDA_VISIBLE_DEVICES.")

        gpu_index: Optional[int] = None
        if cfg.device_map:
            dm = str(cfg.device_map).strip()
            if dm.startswith("cuda:"):
                try:
                    gpu_index = int(dm.split(":", 1)[1])
                except ValueError:
                    gpu_index = None
        if gpu_index is not None:
            torch.cuda.set_device(gpu_index)
        self._device = device

        load_kwargs: Dict[str, Any] = {}
        attn_impl = str(cfg.attn_implementation).strip() if cfg.attn_implementation is not None else ""
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl
        if cfg.device_map:
            load_kwargs["device_map"] = str(cfg.device_map)

        model = llava.load(
            cfg.model_path,
            model_base=None,
            **{k: v for k, v in load_kwargs.items() if v is not None},
        )

        conv_key = str(cfg.conv_template).strip()
        if conv_key:
            if conv_key not in clib.conv_templates:
                raise ValueError(f"Unknown conv_template: {conv_key}")
            clib.default_conversation = clib.conv_templates[conv_key].copy()

        _configure_ps3_and_context_length(model)
        model.eval()
        self._model = model

    def name(self) -> str:
        return self._cfg.name

    def predict(self, item: QAItem, *, frames_data_urls: Optional[Sequence[str]], prompt: str) -> Prediction:
        import torch
        from PIL import Image
        from transformers import GenerationConfig

        frames: list[Image.Image] = []
        if frames_data_urls:
            for u in frames_data_urls:
                raw, _mime = _decode_data_url_image(u)
                frames.append(Image.open(io.BytesIO(raw)).convert("RGB"))

        try:
            base_gen = self._model.default_generation_config
        except Exception:
            base_gen = GenerationConfig()
        gen_cfg = copy.deepcopy(base_gen)
        if int(self._cfg.max_new_tokens) > 0:
            gen_cfg.max_new_tokens = int(self._cfg.max_new_tokens)
        if float(self._cfg.temperature) > 0:
            gen_cfg.do_sample = True
            gen_cfg.temperature = float(self._cfg.temperature)
            gen_cfg.top_p = float(self._cfg.top_p)
        else:
            gen_cfg.do_sample = False

        with torch.inference_mode():
            prompt_parts = [*frames, prompt]
            text = self._model.generate_content(
                prompt_parts,
                generation_config=gen_cfg,
                response_format=None,
            ).strip()
        ans = _parse_answer_letter(text)
        return Prediction(
            answer=ans,
            raw=text,
            meta={
                "qa_id": item.id,
                "model_path": self._cfg.model_path,
                "vila_root": self._cfg.vila_root,
                "device": self._device,
                "torch_dtype": self._cfg.torch_dtype,
                "transformers_cache": os.environ.get("TRANSFORMERS_CACHE") or os.environ.get("HF_HOME") or "",
            },
        )


def build_from_config(cfg: dict) -> VILANVILAAdapter:
    name = str(cfg.get("name", "NVILA")).strip()
    model_path = str(cfg.get("model_path", "")).strip()
    vila_root = str(cfg.get("vila_root", "")).strip()
    if not model_path:
        raise ValueError("Missing `model_path` in model config.")
    if not vila_root:
        raise ValueError("Missing `vila_root` in model config.")
    media = str(cfg.get("media", "videos")).strip().lower() or "videos"
    if media not in {"images", "videos"}:
        raise ValueError(f"Unsupported media: {media} (expected images|videos)")

    return VILANVILAAdapter(
        VILANVILAConfig(
            name=name,
            model_path=model_path,
            vila_root=vila_root,
            model_name=str(cfg.get("model_name", "nvila-8b")).strip() or "nvila-8b",
            conv_template=str(cfg.get("conv_template", "vicuna_v1")).strip() or "vicuna_v1",
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
