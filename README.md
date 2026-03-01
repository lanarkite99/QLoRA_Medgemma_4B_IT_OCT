# Retinal OCT Triage Inference

This repo provides inference code for retinal OCT classification and triage report generation using:
- Base model: `google/medgemma-4b-it`
- LoRA adapter: hosted on Hugging Face (default: `meetsiddhapura/2000-chkpnt`)
- LoRA adapters on Hugging Face: https://huggingface.co/lanarkite99/medgemma_lora

## Hardware Requirements

- NVIDIA GPU with CUDA is required.
- Tested setup: NVIDIA Tesla T4 (16 GB VRAM).
- Other CUDA GPUs may work if VRAM is sufficient.
- CPU-only inference is not officially supported.

## Why GPU Is Required

This pipeline uses a multimodal 4B model with vision inputs. Even with 4-bit quantization, CPU-only runs are typically too slow or memory-heavy for practical use.

## Setup

```bash
huggingface-cli login
pip install -r requirements.txt
```

## Run Inference

```bash
python inference.py --image path/to/oct.jpeg --adapter_repo meetsiddhapura/2000-chkpnt
```

Optional arguments:
- `--model_id` (default: `google/medgemma-4b-it`)
- `--hf_token` (or set `HF_TOKEN` env var)

## Troubleshooting

### Error: CUDA GPU is required

If you see `CUDA GPU is required...`:
- Run on a CUDA-enabled machine (local NVIDIA GPU, cloud GPU, Kaggle, Colab, etc.).
- Install a CUDA-compatible PyTorch build.
- Verify GPU visibility:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO_GPU')"
```

## Notes

- Adapter weights are loaded directly from Hugging Face, so they are not stored in this GitHub repo.
- Make sure your HF account has access to required model repos.
