"""
MARL Training Entry Point.

Pipeline:
    1. DataPipeline    → load CSVs, fragment, encode molecules
    2. TrainingRunner   → init model, create envs, train, evaluate, save
"""

import os
import warnings
import json
import argparse
import subprocess

import torch

from pipeline import DataPipeline, TrainingRunner
from utils.custom_loggers import FileSystemLogger
from utils.train_utils import start_rest_uwsgi, wait_for_rest_api

# Suppress specific warning
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.core")

# Single-threaded torch for parallelization via SubprocVecEnv
torch.set_num_threads(1)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train MARL model with DEEPFMPO framework")
    parser.add_argument('--config', type=str, default='configs/new_test.json', help='Path to the config JSON file')
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)

    cfg['name'] = os.path.basename(args.config).split('.')[0]
    logger = FileSystemLogger(cfg, cfg["logdir"])
    logger.info("Starting the MARL training with the DEEPFMPO framework")

    # ─── Data Pipeline ─────────────────────────────────────────
    dp = DataPipeline(cfg, backend='legacy')
    data = dp.run()

    cfg['deepfmpo']['MAX_FRAGMENTS'] = data.max_frag
    cfg['deepfmpo']['MAX_SWAP'] = data.num_bits
    cfg['deepfmpo']['freeze_encodings'] = data.freeze_encodings

    logger.info("Dataset: {} train, {} eval molecules".format(len(data.train_codes), len(data.eval_codes)))
    logger.info("Agents can swap {} bits".format(data.num_bits))
    if data.freeze_encodings:
        logger.info("Frozen scaffold: {} fragment codes protected".format(len(data.freeze_encodings)))

    # ─── Scorer Setup ──────────────────────────────────────────
    prop_types = cfg['properties'].get('types', [None] * len(cfg['properties']['names']))
    needs_rest = cfg['properties']['use_scorer'] or any(t is not None for t in prop_types)
    rest_process = None

    if needs_rest:
        logger.info("Starting scoring function REST API...")
        my_scorer = 'rest'
        rest_process = start_rest_uwsgi()
        wait_for_rest_api(url="http://localhost:2000", timeout=30)
    else:
        warnings.warn("No scoring function. Agents evaluated on RDKit properties only.")
        my_scorer = None

    # ─── Training ──────────────────────────────────────────────
    runner = TrainingRunner(
        cfg=cfg,
        train_mols=data.train_codes,
        eval_mols=data.eval_codes,
        decodings=data.decodings,
        train_logger=logger,
        scorer=my_scorer,
    )
    runner.run()

    # ─── Cleanup ───────────────────────────────────────────────
    if rest_process is not None:
        logger.info("Shutting down REST API...")
        import signal
        rest_process.send_signal(signal.SIGTERM)
        try:
            rest_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            rest_process.terminate()
