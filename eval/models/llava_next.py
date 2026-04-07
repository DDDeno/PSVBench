from __future__ import annotations

import base64
import copy
import io
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

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


@dataclass(frozen=True)
class LLaVANeXTConfig:
    name: str
    model_path: str
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
    video_fps: float = 1.0
    video_force_sample: bool = True
    default_num_frames: int = 32
    add_video_time_instruction: bool = True
    video_sampling_mode: str = "native"


class LLaVANeXTAdapter(ModelAdapter):
    """
    Adapter for LLaVA-NeXT / LLaVA-Video checkpoints that require the upstream `llava` codebase.

    This is needed for models like `lmms-lab/LLaVA-Video-7B-Qwen2`, where the HF `model_type`
    is `llava` but the actual architecture is Qwen2-based (vocab/hidden-size differ from LLaMA).
    """

    def __init__(self, cfg: LLaVANeXTConfig) -> None:
        self._cfg = cfg

        import torch

        if cfg.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = cfg.device

        if cfg.torch_dtype == "auto":
            resolved_torch_dtype = None
            llava_torch_dtype = None
        elif cfg.torch_dtype == "float32":
            resolved_torch_dtype = torch.float32
            llava_torch_dtype = None
        elif cfg.torch_dtype == "float16":
            resolved_torch_dtype = torch.float16
            llava_torch_dtype = "float16"
        elif cfg.torch_dtype == "bfloat16":
            resolved_torch_dtype = torch.bfloat16
            llava_torch_dtype = "bfloat16"
        else:
            raise ValueError(f"Unsupported torch_dtype: {cfg.torch_dtype}")

        self._device = device

        # LLaVA-NeXT currently imports `apply_chunking_to_forward` from
        # `transformers.modeling_utils`, but newer transformers moved it elsewhere.
        # Patch it in (best-effort) so we can run with modern transformers.
        try:
            import transformers  # noqa: F401
            import transformers.modeling_utils as modeling_utils  # type: ignore

            from transformers import pytorch_utils as _pt_utils  # type: ignore

            for _name in (
                "apply_chunking_to_forward",
                "find_pruneable_heads_and_indices",
                "prune_linear_layer",
            ):
                if not hasattr(modeling_utils, _name) and hasattr(_pt_utils, _name):
                    setattr(modeling_utils, _name, getattr(_pt_utils, _name))
        except Exception:
            pass

        # LLaVA's vision tower builder treats absolute paths as CLIP by default, even when the
        # tower is SigLIP (common for local checkpoints). That leads to shape mismatches like
        # 729 vs 730 position embeddings. Patch the builder to route "siglip" towers correctly.
        try:
            import os as _os

            import llava.model.llava_arch as _llava_arch  # type: ignore
            import llava.model.multimodal_encoder.builder as _mm_builder  # type: ignore
        except Exception:
            _llava_arch = None
            _mm_builder = None

        if _mm_builder is not None and not getattr(_mm_builder, "_siglip_builder_patched", False):
            has_siglip_tower = hasattr(_mm_builder, "SigLipVisionTower")
            if has_siglip_tower:
                def _build_vision_tower_patched(vision_tower_cfg, **kwargs):  # type: ignore
                    vision_tower = getattr(
                        vision_tower_cfg,
                        "mm_vision_tower",
                        getattr(vision_tower_cfg, "vision_tower", None),
                    )
                    if vision_tower is None:
                        raise ValueError("Missing `mm_vision_tower` / `vision_tower` in config.")
                    vision_tower = str(vision_tower)

                    if not _os.path.isabs(vision_tower):
                        candidate_paths = [vision_tower]
                        model_root = str(getattr(vision_tower_cfg, "_name_or_path", "") or "").strip()
                        if model_root:
                            candidate_paths.append(_os.path.normpath(_os.path.join(model_root, vision_tower)))

                        found_local = False
                        for candidate in candidate_paths:
                            if _os.path.exists(candidate):
                                vision_tower = candidate
                                found_local = True
                                break

                        if not found_local:
                            # Relative path like "./models/google/siglip-so400m-patch14-384"
                            # does not exist locally — strip leading dirs to recover a
                            # HuggingFace Hub ID (e.g. "google/siglip-so400m-patch14-384")
                            # so that from_pretrained can auto-download it.
                            cleaned = vision_tower.replace("\\", "/").strip("/")
                            parts = cleaned.split("/")
                            # Drop path prefixes (., models, etc.) to find "org/repo"
                            while len(parts) > 2 and not _os.path.exists("/".join(parts)):
                                parts = parts[1:]
                            if len(parts) >= 2:
                                vision_tower = "/".join(parts)

                    use_s2 = getattr(vision_tower_cfg, "s2", False)

                    # Fix: prefer SigLIP even when it's a local absolute path.
                    if "siglip" in vision_tower.lower():
                        return _mm_builder.SigLipVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)

                    is_absolute_path_exists = _os.path.exists(vision_tower)
                    if (
                        is_absolute_path_exists
                        or vision_tower.startswith("openai")
                        or vision_tower.startswith("laion")
                        or "ShareGPT4V" in vision_tower
                    ):
                        if use_s2:
                            return _mm_builder.CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
                        return _mm_builder.CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
                    if vision_tower.startswith("hf:"):
                        return _mm_builder.HFVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
                    if vision_tower in ["imagebind_huge"]:
                        return _mm_builder.ImageBindWrapper(vision_tower, args=vision_tower_cfg, **kwargs)
                    if vision_tower.startswith("open_clip_hub"):
                        return _mm_builder.OpenCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
                    if "mlcd-vit-bigG-patch14" in vision_tower:
                        if use_s2:
                            return _mm_builder.MLCDVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
                        return _mm_builder.MLCDVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

                    raise ValueError(f"Unknown vision tower: {vision_tower}")

                _mm_builder.build_vision_tower = _build_vision_tower_patched  # type: ignore[attr-defined]
                if _llava_arch is not None:
                    _llava_arch.build_vision_tower = _build_vision_tower_patched  # type: ignore[attr-defined]
                _mm_builder._siglip_builder_patched = True

        try:
            from llava.model.builder import load_pretrained_model  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Missing LLaVA-NeXT runtime. Install it with:\n"
                "  pip install -U git+https://github.com/LLaVA-VL/LLaVA-NeXT.git\n"
                "Then retry."
            ) from e

        load_kwargs: Dict[str, Any] = {
            "device_map": str(cfg.device_map) if device == "cuda" else None,
        }
        if llava_torch_dtype is not None:
            load_kwargs["torch_dtype"] = llava_torch_dtype
        attn_impl = str(cfg.attn_implementation).strip() if cfg.attn_implementation is not None else ""
        if device != "cuda" and (not attn_impl or attn_impl.startswith("flash_attention")):
            attn_impl = "sdpa"
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl

        try:
            # load_pretrained_model(pretrained, model_base, model_name, **llava_model_args)
            import pdb

            orig_set_trace = pdb.set_trace
            pdb.set_trace = lambda *args, **kwargs: None
            try:
                tokenizer, model, image_processor, _max_length = load_pretrained_model(
                    cfg.model_path,
                    None,
                    str(cfg.model_name),
                    **{k: v for k, v in load_kwargs.items() if v is not None},
                )
            finally:
                pdb.set_trace = orig_set_trace
        except Exception as e:
            msg = str(e)
            if "apply_chunking_to_forward" in msg:
                raise RuntimeError(
                    "Your `llava` package is incompatible with the installed `transformers`.\n"
                    "Fix by using a compatible environment, e.g. pin transformers to an older version "
                    "(LLaVA-NeXT historically targets ~4.40), or install a newer LLaVA-NeXT revision "
                    "that supports your transformers.\n"
                    "If you just want a HF-native LLaVA video model, try `bench/eval/configs/models/llava_next_video_7b_local.yaml`."
                ) from e
            raise
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        self._image_processor = image_processor

    def name(self) -> str:
        return self._cfg.name

    def _build_user_text(
        self,
        *,
        prompt: str,
        include_image_token: bool,
        time_instruction: Optional[str] = None,
    ) -> str:
        from llava.constants import DEFAULT_IMAGE_TOKEN  # type: ignore

        system_msg = (
            "You are a strict multiple-choice evaluator. "
            "Reply with only a single letter: A, B, C, or D."
        )
        parts = []
        if include_image_token:
            parts.append(DEFAULT_IMAGE_TOKEN)
        if time_instruction:
            parts.append(time_instruction)
        parts.append(system_msg)
        parts.append(prompt)
        return "\n".join(parts)

    def _tokenize_prompt(self, user_text: str) -> Any:
        from llava.constants import IMAGE_TOKEN_INDEX  # type: ignore
        from llava.conversation import conv_templates  # type: ignore
        from llava.mm_utils import tokenizer_image_token  # type: ignore

        conv = copy.deepcopy(conv_templates[str(self._cfg.conv_template)])
        conv.append_message(conv.roles[0], user_text)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        # For include_image_token==False, this is equivalent to regular tokenization.
        input_ids = tokenizer_image_token(
            prompt_text,
            self._tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0)
        return input_ids

    def _load_video_official(
        self,
        *,
        video_path: str,
        max_frames_num: int,
        fps: float,
        force_sample: bool,
    ) -> tuple[Any, str, float, int]:
        import numpy as np
        from decord import VideoReader, cpu

        if max_frames_num <= 0:
            max_frames_num = 1
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        if total_frame_num <= 0:
            raise ValueError(f"Empty video: {video_path}")

        video_time = total_frame_num / vr.get_avg_fps()
        stride = max(int(round(vr.get_avg_fps() / fps)), 1)
        frame_idx = list(range(0, total_frame_num, stride))
        frame_time = [i / stride for i in frame_idx]
        if len(frame_idx) > max_frames_num or force_sample:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i / vr.get_avg_fps() for i in frame_idx]

        video = vr.get_batch(frame_idx).asnumpy()
        frame_time_str = ",".join(f"{t:.2f}s" for t in frame_time)
        return video, frame_time_str, float(video_time), len(frame_idx)

    def _load_video_from_bench_frames(self, *, frames_data_urls: Sequence[str]) -> tuple[Any, int]:
        from PIL import Image

        frames = []
        for u in frames_data_urls:
            raw, _mime = _decode_data_url_image(u)
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            frames.append(np.array(image))
        if not frames:
            raise ValueError("No bench-sampled frames provided.")
        return np.stack(frames, axis=0), len(frames)

    def predict(self, item: QAItem, *, frames_data_urls: Optional[Sequence[str]], prompt: str) -> Prediction:
        import torch
        from PIL import Image

        sampling_mode = str(self._cfg.video_sampling_mode).strip().lower() or "native"
        if sampling_mode not in {"native", "uniform_bench"}:
            raise ValueError(
                f"Unsupported video_sampling_mode: {self._cfg.video_sampling_mode} "
                "(expected native|uniform_bench)"
            )

        use_video_input = (
            self._cfg.media == "videos"
            and frames_data_urls is not None
            and os.path.exists(item.video_path)
        )
        images = []
        if frames_data_urls and not use_video_input:
            for u in frames_data_urls:
                raw, _mime = _decode_data_url_image(u)
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                images.append(img)

        gen_kwargs: Dict[str, Any] = {"max_new_tokens": int(self._cfg.max_new_tokens)}
        if float(self._cfg.temperature) > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(self._cfg.temperature)
            gen_kwargs["top_p"] = float(self._cfg.top_p)
        else:
            gen_kwargs["do_sample"] = False

        input_mode = "text_only"
        with torch.inference_mode():
            if use_video_input:
                if sampling_mode == "native":
                    max_frames_num = len(frames_data_urls) if frames_data_urls else int(self._cfg.default_num_frames)
                    clip, frame_time, video_time, sampled_frames = self._load_video_official(
                        video_path=item.video_path,
                        max_frames_num=max_frames_num,
                        fps=float(self._cfg.video_fps),
                        force_sample=bool(self._cfg.video_force_sample),
                    )
                    input_mode = "llava_official_video_reader"
                else:
                    clip, sampled_frames = self._load_video_from_bench_frames(frames_data_urls=frames_data_urls or [])
                    frame_time = ""
                    video_time = 0.0
                    input_mode = "llava_bench_sampled_video"
                time_instruction = None
                if bool(self._cfg.add_video_time_instruction):
                    if sampling_mode == "native":
                        time_instruction = (
                            f"The video lasts for {video_time:.2f} seconds, and {sampled_frames} frames "
                            f"are uniformly sampled from it. These frames are located at {frame_time}. "
                            "Please answer the following multiple-choice question about this video."
                        )
                    else:
                        time_instruction = (
                            f"{sampled_frames} frames are provided from the benchmark's pre-sampled video clip. "
                            "Please answer the following multiple-choice question about this video."
                        )
                user_text = self._build_user_text(
                    prompt=prompt,
                    include_image_token=True,
                    time_instruction=time_instruction,
                )
                input_ids = self._tokenize_prompt(user_text).to(self._device)
                pixel_values = self._image_processor.preprocess(clip, return_tensors="pt")["pixel_values"]
                pixel_values = pixel_values.to(device=self._device)
                try:
                    pixel_values = pixel_values.to(dtype=next(self._model.parameters()).dtype)
                except Exception:
                    pass
                videos = [pixel_values]
                out_ids = self._model.generate(
                    input_ids,
                    images=videos,
                    modalities=["video"],
                    **gen_kwargs,
                )
            elif images:
                user_text = self._build_user_text(prompt=prompt, include_image_token=True)
                input_ids = self._tokenize_prompt(user_text).to(self._device)
                # Multi-image path: feed frames as a list of images.
                from llava.mm_utils import process_images  # type: ignore

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
                input_mode = "multi_image"
                out_ids = self._model.generate(input_ids, images=images_arg, **gen_kwargs)
            else:
                user_text = self._build_user_text(prompt=prompt, include_image_token=False)
                input_ids = self._tokenize_prompt(user_text).to(self._device)
                out_ids = self._model.generate(input_ids, **gen_kwargs)

        prompt_len = int(input_ids.shape[1])
        new_ids = out_ids[:, prompt_len:] if prompt_len else out_ids
        if hasattr(new_ids, "shape") and int(new_ids.shape[1]) == 0:
            new_ids = out_ids

        text = self._tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
        if not text:
            text = self._tokenizer.batch_decode(new_ids, skip_special_tokens=False)[0].strip()
        ans = _parse_answer_letter(text)
        return Prediction(
            answer=ans,
            raw=text,
            meta={
                "qa_id": item.id,
                "model_path": self._cfg.model_path,
                "device": self._device,
                "torch_dtype": self._cfg.torch_dtype,
                "video_sampling_mode": sampling_mode,
                "transformers_cache": os.environ.get("TRANSFORMERS_CACHE") or os.environ.get("HF_HOME") or "",
                "input_mode": input_mode,
            },
        )


def build_from_config(cfg: dict) -> LLaVANeXTAdapter:
    name = str(cfg.get("name", "llava_next")).strip()
    model_path = str(cfg.get("model_path", "")).strip()
    if not model_path:
        raise ValueError("Missing `model_path` in model config.")
    media = str(cfg.get("media", "videos")).strip().lower() or "videos"
    if media not in {"images", "videos"}:
        raise ValueError(f"Unsupported media: {media} (expected images|videos)")
    return LLaVANeXTAdapter(
        LLaVANeXTConfig(
            name=name,
            model_path=model_path,
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
            video_fps=float(cfg.get("video_fps", 1.0)),
            video_force_sample=bool(cfg.get("video_force_sample", True)),
            default_num_frames=int(cfg.get("default_num_frames", 32)),
            add_video_time_instruction=bool(cfg.get("add_video_time_instruction", True)),
            video_sampling_mode=str(cfg.get("video_sampling_mode", "native")).strip() or "native",
        )
    )
