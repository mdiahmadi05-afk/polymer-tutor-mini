# Polymer Tutor Mini

A Persian instruction-tuned language model for educational question
answering in polymer science.

## Project Status

🚧 Work in Progress

## Project Goals

- Answer polymer science questions in Persian
- Explain scientific concepts step by step
- Preserve important English terminology
- Reduce unsupported and fabricated responses
- Compare the base model with the fine-tuned model

## Initial Scope

The first version focuses on physical chemistry of polymers.

## Planned Method

- Base model: To be selected
- Fine-tuning method: QLoRA
- Dataset format: Chat-based JSONL
- Evaluation: Baseline vs fine-tuned model
- Hardware: NVIDIA RTX 4070 Laptop GPU, 8 GB VRAM

## Repository Structure

```text
data/       Dataset samples and processing outputs
src/        Training, inference, and evaluation scripts
configs/    Training configurations
tests/      Dataset and code tests
docs/       Project documentation
results/    Evaluation results