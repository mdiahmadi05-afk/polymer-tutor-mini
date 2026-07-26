# Polymer Tutor Mini — Experiment Status

## Current status

The Persian polymer tutor was tested using Gemma 3 4B with QLoRA.

### Available data

- 90 positive scientific training examples
- 30 insufficient-context examples
- 15 untouched holdout examples
- Rebalanced dataset:
  - 82 positive training examples
  - 8 insufficient-context training examples
  - 10 validation examples

### QLoRA results

Two main adapters were tested:

1. `outputs/qlora_verified_120_run1`
2. `outputs/qlora_rebalanced_100_v2`

Observed problems:

- Excessive refusal even when the answer was present in the context
- Incorrect polymer-science conclusions
- Incorrect arithmetic
- Capability degradation compared with the base model
- Memorization of response templates instead of reliable grounding

The current adapters must not be used in the final application.

## Technical conclusion

Small template-heavy SFT datasets are not sufficient for simultaneously teaching:

- Polymer science
- Context-support detection
- Scientific reasoning
- Numerical calculations
- Correct refusal behavior
- Persian answer generation

## Next architecture

Continue with:

1. Hybrid RAG retrieval
2. Evidence-support classification
3. Evidence extraction
4. Python-based numerical calculations
5. Base Gemma model for Persian answer generation
6. Final answer verification and source citation

Use Odian and Sperling first. Add more books only after the pipeline works reliably.

## Next step

Build and evaluate the source-grounded RAG pipeline before attempting further fine-tuning.
