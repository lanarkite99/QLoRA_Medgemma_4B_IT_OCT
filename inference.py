import argparse
import os

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

CLASSES = ["CNV", "DME", "DRUSEN", "NORMAL"]
DIAGNOSIS_MAP = {
    "CNV": "Choroidal Neovascularization",
    "DME": "Diabetic Macular Edema",
    "DRUSEN": "Drusen",
    "NORMAL": "No significant abnormality detected",
}
URGENCY_MAP = {
    "CNV": "High",
    "DME": "Moderate",
    "DRUSEN": "Low",
    "NORMAL": "Low",
}


def _build_quant_config() -> BitsAndBytesConfig:
    # Kaggle T4 is compute capability 7.5 and works best with float16 compute.
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def load_model(model_id: str, adapter_repo: str, hf_token: str | None = None):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required. This pipeline is configured for T4-style GPU inference."
        )

    quant_config = _build_quant_config()
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=hf_token,
    )
    base = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
        token=hf_token,
    )
    model = PeftModel.from_pretrained(base, adapter_repo, token=hf_token)
    model.eval()
    return model, processor


def _generate(
    model,
    processor,
    image: Image.Image,
    system_text: str,
    user_text: str,
    max_new_tokens: int,
) -> str:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image", "image": image},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    generation = outputs[0][input_len:]
    return processor.decode(generation, skip_special_tokens=True).strip()


def normalize_prediction(response: str) -> str:
    parsed = response.strip().upper()
    for cls in CLASSES:
        if cls in parsed:
            return cls
    return "UNKNOWN"


def classify_oct(model, processor, image: Image.Image) -> str:
    raw = _generate(
        model=model,
        processor=processor,
        image=image,
        system_text="You are a retinal OCT classification expert.",
        user_text="Classify this OCT scan. Output exactly one word: CNV, DME, DRUSEN, NORMAL.",
        max_new_tokens=5,
    )
    return normalize_prediction(raw)


def generate_triage_report(model, processor, image: Image.Image, label: str) -> str:
    diagnosis = DIAGNOSIS_MAP.get(label, label)
    urgency = URGENCY_MAP.get(label, "Moderate")

    prompt = f"""The OCT scan has already been classified as: {diagnosis}.

Generate output using EXACT headings:
Diagnosis:
Clinical Summary:
Urgency Level:
Recommended Action:
Patient Explanation:

Rules:
- Diagnosis must be exactly: {diagnosis}
- Urgency Level must be exactly: {urgency}
- Keep concise and factual.
"""

    return _generate(
        model=model,
        processor=processor,
        image=image,
        system_text="You are a cautious retinal specialist generating a structured OCT triage note.",
        user_text=prompt,
        max_new_tokens=180,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="OCT classification + triage inference")
    parser.add_argument("--image", required=True, help="Path to OCT image")
    parser.add_argument("--model_id", default="google/medgemma-4b-it")
    parser.add_argument("--adapter_repo", default="meetsiddhapura/2000-chkpnt")
    parser.add_argument("--hf_token", default=os.getenv("HF_TOKEN"))
    args = parser.parse_args()

    model, processor = load_model(
        model_id=args.model_id,
        adapter_repo=args.adapter_repo,
        hf_token=args.hf_token,
    )

    image = Image.open(args.image).convert("RGB")
    label = classify_oct(model, processor, image)
    report = generate_triage_report(model, processor, image, label)

    print(f"Predicted Label: {label}\n")
    print("=== TRIAGE REPORT ===")
    print(report)


if __name__ == "__main__":
    main()
