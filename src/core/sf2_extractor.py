# sf2_extractor.py
#
# SB3 features extractor for the compact V4 observation. Continuous features and
# binary flags pass through; categorical IDs go through nn.Embedding.

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

ACT_CATEGORIES = 256
CHAR_CATEGORIES = 16


class SF2FeaturesExtractor(BaseFeaturesExtractor):
    """Embeds the categorical IDs of a stacked V4 observation.

    Per frame the tail is laid out as:
        [cont_dim continuous][flag_dim flags]
        [p1_act_hi, p2_act_hi][p1_act_lo, p2_act_lo, p1_btn, p2_btn]
        [p1_char, p2_char]

    The two action high-bytes get the widest embedding because they carry the
    move identity. The low bytes and raw button reads are auxiliary and get a
    narrower one. Characters are a 16-way choice and need very little.
    """

    def __init__(self, observation_space, n_frames: int = 4, frame_dim: int = 23,
                 cont_dim: int = 13, flag_dim: int = 2,
                 act_embed_dim: int = 32, aux_embed_dim: int = 16,
                 char_embed_dim: int = 8):
        features_dim = n_frames * (
            cont_dim + flag_dim
            + 2 * act_embed_dim + 4 * aux_embed_dim + 2 * char_embed_dim
        )
        super().__init__(observation_space, features_dim=features_dim)

        self.n_frames = n_frames
        self.frame_dim = frame_dim
        self.cont_dim = cont_dim
        self.flag_dim = flag_dim
        self._id_start = cont_dim + flag_dim

        self.act_embed = nn.Embedding(ACT_CATEGORIES, act_embed_dim)
        self.aux_embed = nn.Embedding(ACT_CATEGORIES, aux_embed_dim)
        self.char_embed = nn.Embedding(CHAR_CATEGORIES, char_embed_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch = observations.shape[0]
        frames = observations.view(batch, self.n_frames, self.frame_dim)

        passthrough = frames[:, :, :self._id_start]               # (B, F, 15)
        ids = frames[:, :, self._id_start:].round().long()        # (B, F, 8)

        # A corrupt payload or an unseen ROM state must not index out of bounds
        # and kill a 16-worker training run.
        act_ids = ids[:, :, 0:2].clamp(0, ACT_CATEGORIES - 1)
        aux_ids = ids[:, :, 2:6].clamp(0, ACT_CATEGORIES - 1)
        char_ids = ids[:, :, 6:8].clamp(0, CHAR_CATEGORIES - 1)

        act_vec = self.act_embed(act_ids).flatten(start_dim=2)     # (B, F, 2*32)
        aux_vec = self.aux_embed(aux_ids).flatten(start_dim=2)     # (B, F, 4*16)
        char_vec = self.char_embed(char_ids).flatten(start_dim=2)  # (B, F, 2*8)

        return torch.cat(
            [passthrough, act_vec, aux_vec, char_vec], dim=2
        ).flatten(start_dim=1)
