from __future__ import annotations

import base64
import io
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from eval.models.base import ModelAdapter, Prediction
from eval.qa_schema import QAItem


_ANS_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def _parse_answer_letter(text: str) -> Optional[str]:
    if not text:
        return None
    normalized = text.strip().upper()
    if normalized in {"A", "B", "C", "D"}:
        return normalized
    match = _ANS_RE.search(normalized)
    if match is None:
        return None
    return match.group(1).upper()


@dataclass(frozen=True)
class InternVLChatOfficialConfig:
    name: str
    model_path: str
    device: str = "auto"
    torch_dtype: str = "bfloat16"
    media: str = "videos"
    trust_remote_code: bool = True
    fix_mistral_regex: bool = True
    attn_implementation: Optional[str] = None
    max_new_tokens: int = 16
    temperature: float = 0.0
    top_p: float = 1.0
    input_size: int = 448
    video_max_num: int = 1
    default_num_segments: int = 32
    video_sampling_mode: str = "native"


class InternVLChatOfficialAdapter(ModelAdapter):
    def __init__(self, cfg: InternVLChatOfficialConfig) -> None:
        self._cfg = cfg

        import torch
        from transformers import AutoModel, AutoTokenizer

        if cfg.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = cfg.device
        self._device = device

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

        tokenizer_kwargs: Dict[str, Any] = {
            "trust_remote_code": bool(cfg.trust_remote_code),
            "use_fast": False,
        }
        if bool(cfg.fix_mistral_regex):
            tokenizer_kwargs["fix_mistral_regex"] = True
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, **tokenizer_kwargs)
        except TypeError:
            tokenizer_kwargs.pop("fix_mistral_regex", None)
            self._tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, **tokenizer_kwargs)

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": bool(cfg.trust_remote_code),
            "low_cpu_mem_usage": True,
        }
        if resolved_torch_dtype is not None:
            model_kwargs["torch_dtype"] = resolved_torch_dtype
        if device == "cuda":
            model_kwargs["device_map"] = "auto"
        if cfg.attn_implementation and str(cfg.attn_implementation).startswith("flash_attention"):
            model_kwargs["use_flash_attn"] = True

        self._model = AutoModel.from_pretrained(cfg.model_path, **model_kwargs).eval()
        if device != "cuda":
            self._model.to(device)

    def name(self) -> str:
        return self._cfg.name

    def _build_transform(self, *, input_size: int):
        from torchvision import transforms as T
        from torchvision.transforms.functional import InterpolationMode

        return T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _find_closest_aspect_ratio(
        self,
        *,
        aspect_ratio: float,
        target_ratios: list[tuple[int, int]],
        width: int,
        height: int,
        image_size: int,
    ) -> tuple[int, int]:
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def _dynamic_preprocess(
        self,
        image: "Image.Image",
        *,
        min_num: int = 1,
        max_num: int = 12,
        image_size: int = 448,
        use_thumbnail: bool = False,
    ) -> list["Image.Image"]:
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = sorted(
            {
                (i, j)
                for n in range(min_num, max_num + 1)
                for i in range(1, n + 1)
                for j in range(1, n + 1)
                if min_num <= i * j <= max_num
            },
            key=lambda x: x[0] * x[1],
        )
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio=aspect_ratio,
            target_ratios=target_ratios,
            width=orig_width,
            height=orig_height,
            image_size=image_size,
        )
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        tiles_per_row = target_width // image_size
        for i in range(blocks):
            box = (
                (i % tiles_per_row) * image_size,
                (i // tiles_per_row) * image_size,
                ((i % tiles_per_row) + 1) * image_size,
                ((i // tiles_per_row) + 1) * image_size,
            )
            processed_images.append(resized_img.crop(box))
        if use_thumbnail and len(processed_images) != 1:
            processed_images.append(image.resize((image_size, image_size)))
        return processed_images

    def _load_video(self, *, video_path: str, num_segments: int):
        import numpy as np
        import torch
        from decord import VideoReader, cpu
        from PIL import Image

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        num_segments = max(int(num_segments), 1)
        seg_size = float(max(max_frame, 0)) / num_segments if max_frame > 0 else 0.0
        if max_frame <= 0:
            frame_indices = np.array([0], dtype=int)
        else:
            frame_indices = np.array(
                [
                    min(max(int((seg_size / 2.0) + np.round(seg_size * idx)), 0), max_frame)
                    for idx in range(num_segments)
                ],
                dtype=int,
            )

        transform = self._build_transform(input_size=int(self._cfg.input_size))
        pixel_values_list = []
        num_patches_list = []
        for frame_index in frame_indices.tolist():
            image = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
            tiles = self._dynamic_preprocess(
                image,
                image_size=int(self._cfg.input_size),
                use_thumbnail=True,
                max_num=int(self._cfg.video_max_num),
            )
            pixel_values = torch.stack([transform(tile) for tile in tiles])
            num_patches_list.append(int(pixel_values.shape[0]))
            pixel_values_list.append(pixel_values)
        return torch.cat(pixel_values_list, dim=0), num_patches_list

    def _build_video_prompt(self, *, prompt: str, num_patches_list: Sequence[int]) -> str:
        frame_prefix = "".join(f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list)))
        return frame_prefix + prompt

    def _load_images(self, *, frames_data_urls: Sequence[str]):
        import torch
        from PIL import Image

        transform = self._build_transform(input_size=int(self._cfg.input_size))
        pixel_values_list = []
        num_patches_list = []
        for data_url in frames_data_urls:
            if not isinstance(data_url, str) or "," not in data_url:
                raise ValueError("Invalid frame data URL.")
            _, b64_payload = data_url.split(",", 1)
            image = Image.open(io.BytesIO(base64.b64decode(b64_payload))).convert("RGB")
            tiles = self._dynamic_preprocess(
                image,
                image_size=int(self._cfg.input_size),
                use_thumbnail=True,
                max_num=int(self._cfg.video_max_num),
            )
            pixel_values = torch.stack([transform(tile) for tile in tiles])
            num_patches_list.append(int(pixel_values.shape[0]))
            pixel_values_list.append(pixel_values)
        return torch.cat(pixel_values_list, dim=0), num_patches_list

    def predict(self, item: QAItem, *, frames_data_urls: Optional[Sequence[str]], prompt: str) -> Prediction:
        import torch

        gen_cfg: Dict[str, Any] = {"max_new_tokens": int(self._cfg.max_new_tokens)}
        if float(self._cfg.temperature) > 0:
            gen_cfg["do_sample"] = True
            gen_cfg["temperature"] = float(self._cfg.temperature)
            gen_cfg["top_p"] = float(self._cfg.top_p)
        else:
            gen_cfg["do_sample"] = False

        sampling_mode = str(self._cfg.video_sampling_mode).strip().lower() or "native"
        if sampling_mode not in {"native", "uniform_bench"}:
            raise ValueError(
                f"Unsupported video_sampling_mode: {self._cfg.video_sampling_mode} "
                "(expected native|uniform_bench)"
            )

        input_mode = "internvl_chat_text_only"
        if self._cfg.media == "videos" and frames_data_urls is not None and os.path.exists(item.video_path):
            if sampling_mode == "native":
                num_segments = int(len(frames_data_urls)) if frames_data_urls else int(self._cfg.default_num_segments)
                pixel_values, num_patches_list = self._load_video(video_path=item.video_path, num_segments=num_segments)
                input_mode = "internvl_chat_official_video_reader"
            else:
                pixel_values, num_patches_list = self._load_images(frames_data_urls=frames_data_urls)
                input_mode = "internvl_chat_bench_sampled_frames"
            pixel_values = pixel_values.to(self._device)
            try:
                pixel_values = pixel_values.to(dtype=next(self._model.parameters()).dtype)
            except Exception:
                pass
            question = self._build_video_prompt(prompt=prompt, num_patches_list=num_patches_list)
            with torch.inference_mode():
                text = self._model.chat(
                    self._tokenizer,
                    pixel_values,
                    question,
                    gen_cfg,
                    history=None,
                    return_history=False,
                    num_patches_list=list(num_patches_list),
                )
        elif frames_data_urls:
            pixel_values, num_patches_list = self._load_images(frames_data_urls=frames_data_urls)
            pixel_values = pixel_values.to(self._device)
            try:
                pixel_values = pixel_values.to(dtype=next(self._model.parameters()).dtype)
            except Exception:
                pass
            question = self._build_video_prompt(prompt=prompt, num_patches_list=num_patches_list)
            input_mode = "internvl_chat_official_images"
            with torch.inference_mode():
                text = self._model.chat(
                    self._tokenizer,
                    pixel_values,
                    question,
                    gen_cfg,
                    history=None,
                    return_history=False,
                    num_patches_list=list(num_patches_list),
                )
        else:
            with torch.inference_mode():
                text = self._model.chat(
                    self._tokenizer,
                    None,
                    prompt,
                    gen_cfg,
                    history=None,
                    return_history=False,
                    num_patches_list=None,
                )

        return Prediction(
            answer=_parse_answer_letter(text),
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


def build_from_config(cfg: dict) -> InternVLChatOfficialAdapter:
    model_path = str(cfg.get("model_path", "")).strip()
    if not model_path:
        raise ValueError("Missing `model_path` in model config.")
    media = str(cfg.get("media", "videos")).strip().lower() or "videos"
    if media not in {"images", "videos"}:
        raise ValueError(f"Unsupported media: {media} (expected images|videos)")
    return InternVLChatOfficialAdapter(
        InternVLChatOfficialConfig(
            name=str(cfg.get("name", "internvl_chat_official")).strip() or "internvl_chat_official",
            model_path=model_path,
            device=str(cfg.get("device", "auto")).strip() or "auto",
            torch_dtype=str(cfg.get("torch_dtype", "bfloat16")).strip() or "bfloat16",
            media=media,
            trust_remote_code=bool(cfg.get("trust_remote_code", True)),
            fix_mistral_regex=bool(cfg.get("fix_mistral_regex", True)),
            attn_implementation=str(cfg.get("attn_implementation", "")).strip() or None,
            max_new_tokens=int(cfg.get("max_new_tokens", 16)),
            temperature=float(cfg.get("temperature", 0.0)),
            top_p=float(cfg.get("top_p", 1.0)),
            input_size=int(cfg.get("input_size", 448)),
            video_max_num=int(cfg.get("video_max_num", 1)),
            default_num_segments=int(cfg.get("default_num_segments", 32)),
            video_sampling_mode=str(cfg.get("video_sampling_mode", "native")).strip() or "native",
        )
    )
