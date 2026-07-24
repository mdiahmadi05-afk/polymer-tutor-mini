from pathlib import Path

import torch
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
)

MODEL_PATH = Path("models/gemma-3-4b-it")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"مدل پیدا نشد: {MODEL_PATH}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA در دسترس نیست.")

print("GPU:", torch.cuda.get_device_name(0))
print("در حال بارگذاری مدل 4-bit...")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    quantization_config=quantization_config,
    dtype=torch.bfloat16,
    device_map={"": 0},
    attn_implementation="sdpa",
    low_cpu_mem_usage=True,
)

model.config.use_cache = False

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    exclude_modules=r".*(vision_tower|multi_modal_projector).*",
)

model = get_peft_model(model, lora_config)

trainable_parameters = [
    name
    for name, parameter in model.named_parameters()
    if parameter.requires_grad
]

vision_trainable = [
    name
    for name in trainable_parameters
    if "vision_tower" in name or "multi_modal_projector" in name
]

trainable_count = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

total_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("\nنتیجه بررسی:")
print(f"پارامترهای قابل‌آموزش: {trainable_count:,}")
print(f"کل پارامترها: {total_count:,}")
print(f"درصد قابل‌آموزش: {100 * trainable_count / total_count:.4f}%")
print(f"پارامتر قابل‌آموزش در بخش بینایی: {len(vision_trainable)}")

if vision_trainable:
    print("\nخطا: تعدادی از پارامترهای بینایی قابل‌آموزش شده‌اند:")
    for name in vision_trainable[:10]:
        print(" ", name)
    raise RuntimeError("تنظیم LoRA باید اصلاح شود.")

allocated = torch.cuda.memory_allocated() / 1024**3
reserved = torch.cuda.memory_reserved() / 1024**3

print(f"حافظه تخصیص‌یافته GPU: {allocated:.2f} GB")
print(f"حافظه رزروشده GPU: {reserved:.2f} GB")
print("\nتنظیم QLoRA سالم است.")
