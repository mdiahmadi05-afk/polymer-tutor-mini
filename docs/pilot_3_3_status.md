# Odian Section 3-3 Pilot Status

## Completed

- Source extraction for Odian Chapter 3, Section 3-3
- Final knowledge records: 49
- Final QA records: 147
- Final dataset audit passed with 0 errors and 0 warnings
- PDF mapping verified:
  - Printed pages 204-209
  - PDF pages 231-236
- Pilot splits:
  - Train: 80
  - In-domain evaluation: 40
  - Challenge validation: 12
  - Challenge holdout: 15
- Concept/source-record leakage: 0
- Gemma 3 4B QLoRA training completed:
  - Global steps: 20
  - Best eval loss: 3.632235
  - LoRA rank: 8
  - Learning rate: 5e-5
  - Best adapter local path:
    outputs/pilot_3_3/qlora_r8_lr5e5/best_adapter

## Important

Models, adapters, checkpoints, book PDFs, generated datasets, and evaluation
outputs are local-only and are not committed to GitHub.

The current evaluation script manually overrides eos_token_id, causing every
answer to reach max_new_tokens. Existing generation metrics must not yet be
used for final model comparison.

## Next step

1. Fix EOS handling in src/evaluate_gemma3_odian_3_3_model.py
2. Run a two-question base/adapter smoke evaluation
3. Re-run in-domain and challenge-validation evaluation
4. Use challenge holdout only for the final untouched evaluation
