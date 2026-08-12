"""Allow running the training CLI via: python -m t0_training / torchrun -m t0_training."""

from dotenv import load_dotenv

load_dotenv()

from t0_training.olmo.cli import train_main

train_main()
