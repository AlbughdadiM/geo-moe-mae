"""
A script to evaluate the MOE-MAE reconstruction results.

Author: Mohanad Albughdadi
Created: 2025-09-12
"""

import argparse
from torchvision import transforms
from eval.evaluate_mae import MAEEvaluator
from transformation.transformer import ToFloat, CenterCrop40, ZScoreNormalize
from utils.data_config import BigEarthNetInfo
from utils.data_utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Evaluating MAE MMLiT")
    parser.add_argument("--config_yaml", type=str, required=True)
    args = parser.parse_args()
    bigearth_transforms = transforms.Compose(
        [
            ToFloat(),
            CenterCrop40(),
            ZScoreNormalize(
                BigEarthNetInfo.STATISTICS["mean"],
                BigEarthNetInfo.STATISTICS["std"],
            ),
        ]
    )
    config = load_config(args.config_yaml)
    evaluator = MAEEvaluator(config, bigearth_transforms)
    metrics = evaluator.run()
    print(f"Masked Loss: {metrics['masked_loss']:.4f}")
    print(f"Unmasked Loss: {metrics['unmasked_loss']:.4f}")
    print(f"MoE Loss: {metrics['moe_loss']:.4f}")
    print(f"Total Loss: {metrics['total_loss']:.4f}")
    print(f"Full Loss: {metrics['full_loss']:.4f}")


if __name__ == "__main__":
    main()
