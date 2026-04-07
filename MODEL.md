# Model Preparation

Model checkpoints are specified as HuggingFace Hub IDs in YAML configs (e.g. `model_path: Qwen/Qwen2.5-VL-7B-Instruct`) and will be **downloaded automatically** on first run to `~/.cache/huggingface/`. No manual download is needed for most models.

Only install the dependencies for the adapter(s) you plan to use. All commands below assume you are in the repo root `PSVBench/`.

## API Models (GPT-4o, Gemini, etc.)

No extra packages needed. Set your API key:

```bash
export OPENAI_API_KEY=...     # GPT-4o / GPT-4o-mini
export GOOGLE_API_KEY=...     # Gemini 2.5 Flash / Pro
```

## Most Models (Qwen-VL, InternVL, LLaVA-OneVision, LLaVA-NeXT-Video, LLaVA-Video, VideoChat-Flash, VideoLLaMA3, etc.)

```bash
pip install torch torchvision transformers numpy decord
```

Checkpoints auto-download from HuggingFace Hub. No extra setup needed.


## mPLUG-Owl3

```bash
pip install transformers==4.46.3    # strict version requirement — use a separate env if needed
```

## LongVA

```bash
git clone https://github.com/EvolvingLMMs-Lab/LongVA.git
cd LongVA && pip install -e . && cd ..
```

> Config uses `longva_root: ./LongVA`. Ensure `from longva.model.builder import load_pretrained_model` works from the repo root.

## VILA 1.5 / NVILA 

```bash
git clone https://github.com/NVlabs/VILA.git
cd VILA && pip install -e . && cd ..    # must pip install — installs s2wrapper and other sub-deps
```

## ViLAMP 

```bash
git clone https://github.com/steven-ccq/ViLAMP.git
cd ViLAMP && pip install -e . && cd ..
```

> Also auto-downloads a CLIP model (`sentence-transformers/clip-ViT-B-32`) at runtime.

## FlexSelect 

```bash
git clone https://github.com/yunzhuzhang0918/flexselect
cd flexselect && pip install -e . && cd ..
pip install qwen-vl-utils
```
