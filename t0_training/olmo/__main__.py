"""Allow running the OLMo-core training CLI via: python -m t0_training.olmo / torchrun -m t0_training.olmo."""

from dotenv import load_dotenv

load_dotenv()

from t0_training.olmo.cli import train_main

train_main()
