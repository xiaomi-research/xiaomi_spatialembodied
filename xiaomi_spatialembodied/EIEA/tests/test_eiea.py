"""Unit/shape tests for the public EIEA encoder and cue corruption.

Run with:
    python -m unittest tests.test_eiea -v

The tiny Mamba-2 model keeps this CPU-only and fast; it exercises the same
sequence-construction / last-K extraction / projection path as the real
mamba-130m backbone.
"""
from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

EIEA_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EIEA_DIR.parent
if str(EIEA_DIR) not in sys.path:
    sys.path.insert(0, str(EIEA_DIR))

from corruption import (
    CorruptionConfig, EVAL_CONDITIONS, apply_corruption, apply_eval_condition,
)
from cue_encoder import EIEAConfig, EIEACueEncoder
from preprocess import make_image_tchw
from eiea_data import boxes_from_row, ensure_depth_uint8, score_answer


def tiny_encoder():
    from transformers import MambaConfig, MambaForCausalLM
    mcfg = MambaConfig(
        hidden_size=32, num_hidden_layers=1, state_size=8, expand=2,
        conv_kernel=4, vocab_size=128,
    )
    mamba = MambaForCausalLM(mcfg)
    cfg = EIEAConfig(
        image_dim=8, depth_dim=6, hidden_size=32, d_vlm=16, num_tor=4,
        bev_channels=2, flow_channels=2, num_box_classes=5, box_features=10,
        max_boxes=10, freeze_mamba=True,
    )
    return EIEACueEncoder(mamba, cfg), cfg


class TestCueEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.enc, cls.cfg = tiny_encoder()
        cls.enc.eval()

    def _inputs(self):
        return dict(
            image_embeds=torch.randn(1, 5, self.cfg.image_dim),
            depth_embeds=torch.randn(1, 3, self.cfg.depth_dim),
            boxes=torch.randn(3, 11),
            bev=torch.rand(2, 32, 32),
            text_ids=torch.randint(0, 128, (1, 4)),
        )

    def test_clean_shape(self):
        out, aux = self.enc(**self._inputs(), return_aux=True)
        self.assertEqual(tuple(out.shape), (1, self.cfg.num_tor, self.cfg.d_vlm))
        self.assertEqual(set(aux["modalities"]), {"text", "image", "depth", "box", "bev"})
        # text 4 + image 5 + depth 3 + PHY 1 + boxes 3 + MOD 1 + BEV 64 + queries 4
        self.assertEqual(aux["seq_len"], 4 + 5 + 3 + 1 + 3 + 1 + 64 + 4)

    def test_missing_cues_shape(self):
        out, aux = self.enc(
            image_embeds=torch.randn(1, 5, self.cfg.image_dim),
            depth_embeds=None, boxes=None, bev=None,
            text_ids=torch.randint(0, 128, (1, 4)), return_aux=True)
        self.assertEqual(tuple(out.shape), (1, self.cfg.num_tor, self.cfg.d_vlm))
        self.assertEqual(set(aux["modalities"]), {"text", "image"})
        self.assertEqual(aux["seq_len"], 4 + 5 + 4)

    def test_empty_box_tensor(self):
        inp = self._inputs()
        inp["boxes"] = torch.zeros(0, 11)
        out, aux = self.enc(**inp, return_aux=True)
        self.assertNotIn("box", aux["modalities"])
        self.assertEqual(tuple(out.shape), (1, self.cfg.num_tor, self.cfg.d_vlm))
        # PHY + BEV (single physical modality, no MOD separator)
        self.assertEqual(aux["seq_len"], 4 + 5 + 3 + 1 + 64 + 4)

    def test_flow_optional(self):
        inp = self._inputs()
        inp["flow"] = torch.rand(2, 32, 32)
        out, aux = self.enc(**inp, return_aux=True)
        self.assertIn("flow", aux["modalities"])
        self.assertEqual(aux["seq_len"],
                         4 + 5 + 3 + 1 + 3 + 1 + 64 + 1 + 64 + 4)

    def test_mamba_frozen_but_projections_trainable(self):
        self.assertFalse(any(p.requires_grad for p in self.enc.midi.mamba.parameters()))
        self.assertTrue(all(p.requires_grad for p in self.enc.midi.image_proj.parameters()))
        self.assertTrue(self.enc.queries.requires_grad)

    def test_backward_reaches_trainable_modules_only(self):
        inp = self._inputs()
        out = self.enc(**inp)
        out.float().sum().backward()
        self.assertIsNotNone(self.enc.midi.image_proj[0].weight.grad)
        self.assertIsNotNone(self.enc.queries.grad)
        self.assertTrue(all(p.grad is None for p in self.enc.midi.mamba.parameters()))
        self.enc.zero_grad(set_to_none=True)

    def test_from_midi_copies_projection_when_width_matches(self):
        import torch.nn as nn

        class FakeMIDI(nn.Module):
            def __init__(self, mamba, img_dim, dep_dim, tor_dim):
                super().__init__()
                self.mamba = mamba
                self.image_proj = nn.Sequential(nn.Linear(img_dim, 32), nn.GELU(),
                                                nn.Linear(32, 32))
                self.depth_proj = nn.Sequential(nn.Linear(dep_dim, 32), nn.GELU(),
                                                nn.Linear(32, 32))
                self.tor_proj = nn.Sequential(nn.Linear(32, tor_dim), nn.GELU(),
                                              nn.Linear(tor_dim, tor_dim))

        mamba = self.enc.midi.mamba
        midi = FakeMIDI(mamba, 8, 6, 64)
        cfg = EIEAConfig(image_dim=8, depth_dim=6, hidden_size=32, d_vlm=64,
                         num_tor=4, bev_channels=2, flow_channels=2,
                         num_box_classes=5, box_features=10, max_boxes=10)
        enc = EIEACueEncoder.from_midi(midi, cfg)
        self.assertTrue(torch.allclose(enc.proj.fc1.weight, midi.tor_proj[0].weight))
        self.assertTrue(torch.allclose(enc.proj.fc2.weight, midi.tor_proj[2].weight))
        # 30B-style mismatch (d_vlm=16) must not crash and must not copy.
        cfg16 = EIEAConfig(image_dim=8, depth_dim=6, hidden_size=32, d_vlm=16,
                           num_tor=4, bev_channels=2, flow_channels=2,
                           num_box_classes=5, box_features=10, max_boxes=10)
        enc16 = EIEACueEncoder.from_midi(midi, cfg16)
        self.assertEqual(tuple(enc16.proj.fc1.weight.shape), (16, 32))

    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "cue.pt")
            self.enc.save_checkpoint(p, extra={"step": 7})
            enc2, ck = EIEACueEncoder.from_checkpoint(p, self.enc.midi.mamba)
            self.assertEqual(ck["step"], 7)
            inputs = self._inputs()
            a, _ = self.enc(**inputs, return_aux=True)
            enc2.eval()
            b, _ = enc2(**inputs, return_aux=True)
            self.assertTrue(torch.allclose(a, b, atol=1e-6))


class TestCorruption(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.boxes = np.concatenate([
            rng.integers(0, 5, (10, 1)),
            rng.normal(0, 5, (10, 10)),
        ], axis=1).astype(np.float32)
        self.bev = rng.random((2, 32, 32)).astype(np.float32)
        self.depth = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)

    def test_does_not_mutate_inputs(self):
        b0, v0, d0 = self.boxes.copy(), self.bev.copy(), self.depth.copy()
        cfg = CorruptionConfig.preset("heavy")
        for seed in range(20):
            apply_corruption(self.boxes, self.bev, self.depth, cfg,
                             random.Random(seed))
        self.assertTrue(np.array_equal(self.boxes, b0))
        self.assertTrue(np.array_equal(self.bev, v0))
        self.assertTrue(np.array_equal(self.depth, d0))

    def test_outputs_are_valid(self):
        cfg = CorruptionConfig.preset("standard")
        for seed in range(50):
            out = apply_corruption(self.boxes, self.bev, self.depth, cfg,
                                   random.Random(seed))
            self.assertIsInstance(out.boxes, np.ndarray)
            self.assertEqual(out.boxes.ndim, 2)
            self.assertEqual(out.boxes.shape[1], 11)
            if out.bev is not None:
                self.assertEqual(out.bev.shape, self.bev.shape)
            if out.depth is not None:
                self.assertEqual(out.depth.shape, self.depth.shape)
                self.assertEqual(out.depth.dtype, np.uint8)

    def test_all_eval_conditions(self):
        for cond in EVAL_CONDITIONS:
            for seed in range(3):
                out = apply_eval_condition(
                    cond, self.boxes, self.bev, self.depth,
                    rng=random.Random(seed))
                self.assertEqual(out.boxes.ndim, 2)
                self.assertEqual(out.boxes.shape[1], 11)
        # explicit absence semantics
        self.assertIsNone(apply_eval_condition("no_bev", self.boxes, self.bev,
                                               self.depth).bev)
        self.assertIsNone(apply_eval_condition("no_depth", self.boxes, self.bev,
                                               self.depth).depth)
        self.assertEqual(apply_eval_condition("no_box", self.boxes, self.bev,
                                              self.depth).boxes.shape[0], 0)
        allgone = apply_eval_condition("no_physical", self.boxes, self.bev, self.depth)
        self.assertEqual(allgone.boxes.shape[0], 0)
        self.assertIsNone(allgone.bev)
        self.assertIsNone(allgone.depth)

    def test_keep_one_is_partial(self):
        cfg = CorruptionConfig(p_keep_one=1.0, p_drop_all=0.0,
                               p_drop_box=0.0, p_drop_bev=0.0, p_drop_depth=0.0,
                               p_box_element_drop=0.0,
                               p_bev_noise=0.0, p_bev_mask=0.0, p_bev_channel_drop=0.0,
                               p_depth_noise=0.0, p_depth_mask=0.0, p_depth_flat=0.0)
        out = apply_corruption(self.boxes, self.bev, self.depth, cfg,
                               random.Random(0))
        present = sum([
            out.boxes.shape[0] > 0,
            out.bev is not None,
            out.depth is not None,
        ])
        self.assertEqual(present, 1)
        self.assertIn("keep_only", out.meta)


class TestDataHelpers(unittest.TestCase):
    def test_boxes_and_depth(self):
        row = {"boxes": [[1, 2.0, 3.0, 0.5, 4.0, 2.0, 1.5, 0.1, 0.0, 0.0, 3.6]],
               "question": "Front view: <image> What is the distance?",
               "answer": "3.6"}
        b = boxes_from_row(row)
        self.assertEqual(b.shape, (1, 11))
        self.assertEqual(boxes_from_row({"boxes": []}).shape, (0, 11))
        d = ensure_depth_uint8(np.zeros((8, 8), dtype=np.float32))
        self.assertEqual(d.shape, (8, 8, 3))
        self.assertEqual(d.dtype, np.uint8)

    def test_scoring(self):
        self.assertEqual(score_answer("A", "A"), 1)
        self.assertEqual(score_answer("The answer is B.", "A"), 0)
        self.assertAlmostEqual(score_answer("12.5 m", "10"), 2.5)
        self.assertIsNone(score_answer("unknown", "10"))

    def test_preprocess_tchw(self):
        img = Image.new("RGB", (100, 50), (128, 128, 128))
        t = make_image_tchw([img], temporal_patch_size=2, target_size=64,
                            patch_size=14)
        self.assertEqual(tuple(t.shape), (2, 3, 64, 64))


if __name__ == "__main__":
    unittest.main()
