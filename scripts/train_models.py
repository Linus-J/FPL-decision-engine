#!/usr/bin/env python
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from projection import minutes_model, points_model

if __name__ == "__main__":
    logging.getLogger().info("Training minutes model...")
    minutes_model.train(save=True)

    logging.getLogger().info("Training points model...")
    points_model.train(save=True)

    logging.getLogger().info("All models trained and saved to models/")
