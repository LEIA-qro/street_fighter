# test_sf2_extractor.py

import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import numpy as np
import torch
from gymnasium import spaces
import pytest

from core.sf2_extractor import SF2FeaturesExtractor
from envs.sf2_v4 import V4_FRAME_DIM, V4_CONT_DIM, V4_FLAG_DIM, V4_ID_DIM

N_FRAMES = 4
OBS_DIM = V4_FRAME_DIM * N_FRAMES


def _space():
    return spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)


def test_v4_frame_layout_adds_up():
    assert V4_CONT_DIM == 13
    assert V4_FLAG_DIM == 2
    assert V4_ID_DIM == 8
    assert V4_FRAME_DIM == V4_CONT_DIM + V4_FLAG_DIM + V4_ID_DIM == 23


def test_extractor_output_dimension_matches_its_declared_features_dim():
    ex = SF2FeaturesExtractor(_space(), n_frames=N_FRAMES)
    # cont 13 + flags 2 + 2 action embeds (32) + 4 aux embeds (16) + 2 char embeds (8)
    assert ex.features_dim == N_FRAMES * (13 + 2 + 2 * 32 + 4 * 16 + 2 * 8) == 636
    out = ex(torch.zeros(5, OBS_DIM))
    assert out.shape == (5, ex.features_dim)


def test_extractor_first_layer_is_far_smaller_than_the_one_hot_version():
    ex = SF2FeaturesExtractor(_space(), n_frames=N_FRAMES)
    # 554*4 = 2216 one-hot dims into a 512-unit layer was 1.13M parameters.
    assert ex.features_dim * 512 < 400_000


def test_extractor_distinguishes_different_action_ids():
    ex = SF2FeaturesExtractor(_space(), n_frames=N_FRAMES)
    a = torch.zeros(1, OBS_DIM)
    b = torch.zeros(1, OBS_DIM)
    b[0, 15] = 17.0    # different p1_act_hi in frame 0
    assert not torch.allclose(ex(a), ex(b))


def test_extractor_reads_the_recovered_low_byte_and_opponent_button():
    """The low byte of 0x804E and P2's input at 0x845E were previously
    discarded entirely. They must actually reach the network."""
    ex = SF2FeaturesExtractor(_space(), n_frames=N_FRAMES)
    base = torch.zeros(1, OBS_DIM)

    lo = base.clone()
    lo[0, 17] = 9.0    # p1_act_lo
    assert not torch.allclose(ex(base), ex(lo))

    btn = base.clone()
    btn[0, 20] = 8.0   # p2_btn
    assert not torch.allclose(ex(base), ex(btn))


def test_extractor_clamps_out_of_range_ids_instead_of_crashing():
    ex = SF2FeaturesExtractor(_space(), n_frames=N_FRAMES)
    obs = torch.zeros(1, OBS_DIM)
    obs[0, 15] = 9999.0   # way past the 256-entry action table
    obs[0, 21] = -5.0     # negative char id
    out = ex(obs)
    assert torch.isfinite(out).all()
