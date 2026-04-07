# PSVBench: A Comprehensive Benchmark for Presentation-Style Video Understanding

PSVBench is a benchmark for evaluating multimodal large language models on **presentation-style video understanding**. It covers 2,030 multiple-choice questions across 853 videos spanning 11 academic and educational domains, testing models on 8 fine-grained reasoning sub-tasks grouped into 3 capability categories.

## Installation

```bash
git clone https://github.com/<your-org>/PSVBench.git
cd PSVBench
pip install -r requirements.txt
```

Core dependencies: Python >= 3.9, PyYAML, tqdm, Pillow, imageio-ffmpeg.

## Data Download

Download the PSVBench dataset from `DDDDeno/PSVBench` on HuggingFace and note the path as your `<data_root>`.

```
<data_root>/
├── qa/
│   └── eval.json                       # 2,030 QA items
└── data/
    ├── acl/                            # 11 domains, 853 videos total
    ├── biology/
    ├── chemistry/
    ├── cvpr/
    ├── eccv/
    ├── iclr/
    ├── icml/
    ├── math/
    ├── medicine/
    ├── neurips/
    └── physics/
        └── <video_name>/
            ├── <video_name>.mp4
            └── transcript/
                └── <video_name>.vtt
```

## Model Preparation

Model checkpoints are specified as HuggingFace Hub IDs in YAML configs (e.g. `model_path: Qwen/Qwen2.5-VL-7B-Instruct`) and will be **downloaded automatically** on first run to `~/.cache/huggingface/`. No manual download is needed for most models.

Some adapters require extra source repos or specific package versions. See **[MODEL.md](MODEL.md)** for per-adapter setup instructions.

## Usage

### Quick Start

```bash
# with 32 frames
bash run_example.sh eval/configs/models/your_model_config.yaml --num-frames 32

# with transcript
bash run_example.sh eval/configs/models/your_model_config.yaml --use-transcript

```

If your dataset is not under the repo root, set `DATA_ROOT`:

```bash
DATA_ROOT=/path/to/psvbench bash run_example.sh eval/configs/models/your_model_config.yaml
```

### Full Command

```bash
python -m eval.run_eval \
    --qa-file /path/to/psvbench/qa/eval.json \
    --data-root /path/to/psvbench \
    --model-config eval/configs/models/your_model_config.yaml \
    --num-frames 32 \
    --use-transcript \
    --resume
```

### Key Arguments

| Argument | Description |
|---|---|
| `--qa-file` | Path to the QA JSON file (required) |
| `--data-root` | Root directory for resolving relative video/transcript paths |
| `--model-config` | Path to model YAML config (required) |
| `--num-frames` | Number of frames to uniformly sample (default: 8) |
| `--max-frames` | Hard upper limit on frames (default: 64) |
| `--image-max-side` | Downscale frames so the longest side ≤ this value |
| `--jpeg-q` | ffmpeg JPEG quality: 2 (best) to 31 (worst) |
| `--no-video` | Text-only mode — no video frames sent to model |
| `--use-transcript` | Load and prepend transcript text to prompts |
| `--resume` | Resume from existing `predictions.jsonl` |
| `--continue-on-error` | Record `invalid` on error instead of stopping |
| `--limit` | Evaluate only the first N items (for debugging) |

### Output

Results are saved to `results/<model_name>/` (or `--output-dir`):

- **`predictions.jsonl`** — per-item predictions (one JSON object per line)
- **`summary.json`** — overall and per-task accuracy breakdown

Example `summary.json`:
```json
{
  "model": "gpt-4o",
  "num_items": 2030,
  "overall": {"correct": 912, "total": 2030, "acc": 0.449, "invalid": 3},
  "per_task_group": [
    {"task": "Alignment", "correct": 280, "total": 597, "acc": 0.469},
    {"task": "Thinking", "correct": 350, "total": 788, "acc": 0.444},
    {"task": "Understanding", "correct": 282, "total": 645, "acc": 0.437}
  ],
  "per_task": [
    {"sub_task": "Counterfactual Projection", "correct": 95, "total": 214, "acc": 0.444},
    ...
  ]
}
```



## Directory Structure

```
PSVBench/                               # repo root — always run from here
├── README.md
├── requirements.txt
├── run_example.sh
│
├── eval/                               # evaluation framework
│   ├── run_eval.py                     # main entry point
│   ├── data_loader.py                  # data loading & prompt construction
│   ├── qa_schema.py                    # QA item schema
│   ├── frame_sampler.py                # video frame extraction (ffmpeg)
│   ├── models/                         # model adapters
│   │   ├── base.py                     # abstract base class
│   │   ├── registry.py                 # model factory
│   │   ├── random_choice.py            # random baseline
│   │   ├── api_models.py              # OpenAI / Gemini / Claude
│   │   ├── hf_transformers.py         # HuggingFace transformers
│   │   ├── internvl_chat.py           # InternVL
│   │   ├── llava_next.py             # LLaVA-NeXT / LLaVA-Video / LLaVA-OneVision
│   │   ├── longva.py                  # LongVA
│   │   ├── vila.py                    # VILA / NVILA
│   │   ├── vilamp.py                  # ViLAMP
│   │   ├── videochat_flash.py         # VideoChat-Flash
│   │   ├── mplug_owl3.py             # mPLUG-Owl3
│   │   └── flexselect.py             # FlexSelect
│   └── configs/models/                 # YAML configs for 35+ models
│
├── LongVA/                             # cloned source repos (adapter-specific)
├── VILA/
├── ViLAMP/
└── flexselect/
```

## License

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

## Citation
TBD 
<!-- ```bibtex
@article{psvbench,
  title={PSVBench: A Benchmark for Presentation Slide Video Understanding},
  author={TBD},
  year={2025}
}
``` -->
