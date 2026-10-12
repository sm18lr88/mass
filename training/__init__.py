# Training package: torch-free SFT helpers (sft_*.py) plus GPU scripts run by path.
# Scripts executed as `python training/<script>.py` append the repository root to sys.path and import
# helpers as `training.<module>`, so scripts and tests share one module object (e.g. sft_prompts.SENT).
