
from config import load_config
from src.tracker.imu_tracker_runner import ImuTrackerRunner
from src.utils.logging import logging


if __name__ == "__main__":

    config = load_config("config.yaml")
    logging.info("Running")
    trackerRunner = ImuTrackerRunner(config)
    logging.info("model loaded")
    trackerRunner.async_run_tracker()

