"""Testes rápidos, sem treino nem GPU: python -m unittest discover -s tests."""

import unittest

import numpy as np
import torch

from pa1_experiments import (
    calculate_instance_metrics, corrupt_image, encoder_receptive_field,
    merge_stitch, naive_stitch, tile_starts,
)
from pa1_inference import load_model, predict_probabilities


class PA1Tests(unittest.TestCase):
    def test_matching_perfect_and_merged(self):
        truth = np.array([[0, 1, 1], [0, 0, 2], [0, 0, 2]], dtype=np.int32)
        perfect = calculate_instance_metrics(truth, truth)
        merged = calculate_instance_metrics(truth, (truth > 0).astype(np.int32))
        self.assertEqual(perfect["map"], 1.0)
        self.assertLess(merged["map"], perfect["map"])

    def test_last_tile_covers_border(self):
        self.assertEqual(tile_starts(512, 256, 192), [0, 192, 256])

    def test_fusion_recovers_one_object(self):
        first = np.zeros((4, 4), dtype=np.int32)
        second = np.zeros((4, 4), dtype=np.int32)
        first[1:3, 1:4] = 1
        second[1:3, 0:3] = 1
        tiles = [
            {"top": 0, "left": 0, "instances": first},
            {"top": 0, "left": 2, "instances": second},
        ]
        self.assertEqual(naive_stitch(tiles, (4, 6)).max(), 2)
        self.assertEqual(merge_stitch(tiles, (4, 6)).max(), 1)

    def test_receptive_field_same_output_stride(self):
        self.assertEqual(encoder_receptive_field(1), (68, 8))
        self.assertEqual(encoder_receptive_field(2), (100, 8))

    def test_corruption_seed_is_repeatable(self):
        image = np.full((8, 8), 0.5, dtype=np.float32)
        first = corrupt_image(image, "ruido", 2, seed=42)
        second = corrupt_image(image, "ruido", 2, seed=42)
        np.testing.assert_array_equal(first, second)

    def test_checkpoint_infers_odd_image_size(self):
        model, _, device = load_model(
            "checkpoints/small_unet_boundary_watershed_v2.pt", device="cpu"
        )
        image = torch.zeros(1, 1, 9, 13)
        probabilities = predict_probabilities(model, image, device)
        self.assertEqual(probabilities.shape, (3, 9, 13))
        np.testing.assert_allclose(probabilities.sum(axis=0), 1.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
