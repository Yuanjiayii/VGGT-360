# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
from .dpt_head import DPTHead
from .track_modules.base_track_predictor import BaseTrackerPredictor


class TrackHead(nn.Module):
    """
    Track head that uses DPT head to process tokens and BaseTrackerPredictor for tracking.
    The tracking is performed iteratively, refining predictions over multiple iterations.
    """

    def __init__(
        self,
        dim_in,
        patch_size=14,
        features=128,
        iters=4,
        predict_conf=True,
        stride=2,
        corr_levels=7,
        corr_radius=4,
        hidden_size=384,
        is_all_frames=False
    ):
        """
        Initialize the TrackHead module.

        Args:
            dim_in (int): Input dimension of tokens from the backbone.
            patch_size (int): Size of image patches used in the vision transformer.
            features (int): Number of feature channels in the feature extractor output.
            iters (int): Number of refinement iterations for tracking predictions.
            predict_conf (bool): Whether to predict confidence scores for tracked points.
            stride (int): Stride value for the tracker predictor.
            corr_levels (int): Number of correlation pyramid levels
            corr_radius (int): Radius for correlation computation, controlling the search area.
            hidden_size (int): Size of hidden layers in the tracker network.
        """
        super().__init__()

        self.patch_size = patch_size

        # Feature extractor based on DPT architecture
        # Processes tokens into feature maps for tracking
        self.feature_extractor = DPTHead(
            dim_in=dim_in,
            patch_size=patch_size,
            features=features,
            feature_only=True,  # Only output features, no activation
            down_ratio=2,  # Reduces spatial dimensions by factor of 2
            pos_embed=False,
        )

        # Tracker module that predicts point trajectories
        # Takes feature maps and predicts coordinates and visibility
        self.tracker = BaseTrackerPredictor(
            latent_dim=features,  # Match the output_dim of feature extractor
            predict_conf=predict_conf,
            stride=stride,
            corr_levels=corr_levels,
            corr_radius=corr_radius,
            hidden_size=hidden_size,
        )

        self.iters = iters
        self.is_all_frames = is_all_frames

    def forward(self, aggregated_tokens_list, images, patch_start_idx, query_points=None, iters=None):
        """
        Forward pass of the TrackHead.

        Args:
            aggregated_tokens_list (list): List of aggregated tokens from the backbone.
            images (torch.Tensor): Input images of shape (B, S, C, H, W) where:
                                   B = batch size, S = sequence length.
            patch_start_idx (int): Starting index for patch tokens.
            query_points (torch.Tensor, optional): Initial query points to track.
                                                  If None, points are initialized by the tracker.
            iters (int, optional): Number of refinement iterations. If None, uses self.iters.

        Returns:
            tuple:
                - coord_preds (torch.Tensor): Predicted coordinates for tracked points.
                - vis_scores (torch.Tensor): Visibility scores for tracked points.
                - conf_scores (torch.Tensor): Confidence scores for tracked points (if predict_conf=True).
        """
        B, S, _, H, W = images.shape

        # Extract features from tokens
        # feature_maps has shape (B, S, C, H//2, W//2) due to down_ratio=2
        feature_maps = self.feature_extractor(aggregated_tokens_list, images, patch_start_idx)  # [B, S, C, HH, WW]

        # Use default iterations if not specified
        if iters is None:
            iters = self.iters



        # Perform tracking using the extracted features
        if not self.is_all_frames:
            coord_preds, vis_scores, conf_scores = self.tracker(query_points=query_points, fmaps=feature_maps, iters=iters)
        else:
            vis_scores_list = []
            for f_idx in range(S):
                feature_maps_ = torch.cat([feature_maps[:, [f_idx]], feature_maps[:, :f_idx], feature_maps[:, f_idx + 1:]], dim=1)
                coord_preds_, vis_scores_, conf_scores_ = self.tracker(query_points=query_points, fmaps=feature_maps_, iters=iters)  # vis_scores_: [B, S, N]
                vis_scores_ = vis_scores_.mean(2)  # [B, S]
                vis_scores__ = torch.cat([vis_scores_[:, 1: f_idx + 1], vis_scores_[:, [0]], vis_scores_[:, f_idx + 1:]], dim=1)
                vis_scores__[:, f_idx] = -1000.
                vis_scores_list.append(vis_scores__)
            vis_scores = torch.stack(vis_scores_list, dim=1)  # [B, N, N]

        return coord_preds, vis_scores, conf_scores
