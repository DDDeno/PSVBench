from __future__ import annotations

from typing import Any, Dict

from eval.models.base import ModelAdapter


def build_model(model_cfg: Dict[str, Any]) -> ModelAdapter:
    """Build a model adapter from a YAML config dict."""
    adapter = str(model_cfg.get("adapter", "")).strip()
    if not adapter:
        raise ValueError("Model config missing `adapter`.")

    if adapter == "openai":
        from eval.models.api_models import build_openai_from_config
        return build_openai_from_config(model_cfg)

    if adapter == "gemini":
        from eval.models.api_models import build_gemini_from_config
        return build_gemini_from_config(model_cfg)

    if adapter == "random":
        from eval.models.random_choice import RandomChoiceAdapter
        return RandomChoiceAdapter(
            seed=int(model_cfg.get("seed", 0)),
            name=str(model_cfg.get("name", "random")),
        )

    if adapter == "hf_transformers":
        from eval.models.hf_transformers import build_from_config
        return build_from_config(model_cfg)

    if adapter == "internvl_chat":
        from eval.models.internvl_chat import build_from_config
        return build_from_config(model_cfg)

    if adapter == "llava_next":
        from eval.models.llava_next import build_from_config
        return build_from_config(model_cfg)

    if adapter == "longva":
        from eval.models.longva import build_from_config
        return build_from_config(model_cfg)

    if adapter == "vilamp":
        from eval.models.vilamp import build_from_config
        return build_from_config(model_cfg)

    if adapter == "vila":
        from eval.models.vila import build_from_config
        return build_from_config(model_cfg)

    if adapter == "videochat_flash":
        from eval.models.videochat_flash import build_from_config
        return build_from_config(model_cfg)

    if adapter == "mplug_owl3":
        from eval.models.mplug_owl3 import build_from_config
        return build_from_config(model_cfg)

    if adapter == "flexselect":
        from eval.models.flexselect import build_from_config
        return build_from_config(model_cfg)

    raise ValueError(f"Unknown adapter: {adapter}")
