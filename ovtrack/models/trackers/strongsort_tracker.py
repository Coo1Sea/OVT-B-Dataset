# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import numpy as np
import torch
from mmdet.core import bbox_overlaps
from mmengine.structures import InstanceData
from motmetrics.lap import linear_sum_assignment
from torch import Tensor

from ..builder import TRACKERS
from ovtrack.core.bbox import bbox_xyxy_to_cxcyah
from ovtrack.core.track.transforms import imrenormalize
from .sort_tracker import SortTracker


def cosine_distance(x: Tensor, y: Tensor) -> np.ndarray:
    """compute the cosine distance.

    Args:
        x (Tensor): embeddings with shape (N,C).
        y (Tensor): embeddings with shape (M,C).

    Returns:
        ndarray: cosine distance with shape (N,M).
    """
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    y = y / np.linalg.norm(y, axis=1, keepdims=True)
    dists = 1. - np.dot(x, y.T)
    return dists


@TRACKERS.register_module()
class StrongSORTTracker(SortTracker):
    """Tracker for StrongSORT.

    Args:
        obj_score_thr (float, optional): Threshold to filter the objects.
            Defaults to 0.6.
        reid (dict, optional): Configuration for the ReID model.
            - num_samples (int, optional): Number of samples to calculate the
                feature embeddings of a track. Default to None.
            - image_scale (tuple, optional): Input scale of the ReID model.
                Default to (256, 128).
            - img_norm_cfg (dict, optional): Configuration to normalize the
                input. Default to None.
            - match_score_thr (float, optional): Similarity threshold for the
                matching process. Default to 0.3.
            - motion_weight (float, optional): the weight of the motion cost.
                Defaults to 0.02.
        match_iou_thr (float, optional): Threshold of the IoU matching process.
            Defaults to 0.7.
        num_tentatives (int, optional): Number of continuous frames to confirm
            a track. Defaults to 2.
    """

    def __init__(self,
                 obj_score_thr: float = 0.6, 
                 num_samples=None,
                 img_scale=(256, 128),
                 img_norm_cfg=None,
                 match_score_thr=0.3,
                 motion_weight=0.02,
                 match_iou_thr: float = 0.7,
                 num_tentatives: int = 2,
                 init_cfg=None,
                 **kwargs):
        
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.match_score_thr = match_score_thr
        self.obj_score_thr = obj_score_thr
        self.match_iou_thr = match_iou_thr
        self.num_tentatives = num_tentatives

    def update_track(self, id: int, obj: Tuple[Tensor]) -> None:
        """Update a track."""
        for k, v in zip(self.memo_items, obj):
            v = v[None]
            if self.momentums is not None and k in self.momentums:
                m = self.momentums[k]
                self.tracks[id][k] = (1 - m) * self.tracks[id][k] + m * v
            else:
                self.tracks[id][k].append(v)

        if self.tracks[id].tentative:
            if len(self.tracks[id]['bboxes']) >= self.num_tentatives:
                self.tracks[id].tentative = False
        bbox = bbox_xyxy_to_cxcyah(self.tracks[id].bboxes[-1])  # size = (1, 4)
        assert bbox.ndim == 2 and bbox.shape[0] == 1
        bbox = bbox.squeeze(0).cpu().numpy()
        score = float(self.tracks[id].scores[-1].cpu())
        self.tracks[id].mean, self.tracks[id].covariance = self.kf.update(
            self.tracks[id].mean, self.tracks[id].covariance, bbox, score)

    def track(self,
              model,
              bboxes,
              embeds,
              labels,
              frame_id,
              **kwargs):
        """Tracking forward function.

        Args:
            model (nn.Module): MOT model.
            img (Tensor): of shape (T, C, H, W) encoding input image.
                Typically these should be mean centered and std scaled.
                The T denotes the number of key images and usually is 1 in
                SORT method.
            feats (list[Tensor]): Multi level feature maps of `img`.
            data_sample (:obj:`TrackDataSample`): The data sample.
                It includes information such as `pred_det_instances`.
            data_preprocessor (dict or ConfigDict, optional): The pre-process
               config of :class:`TrackDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
            rescale (bool, optional): If True, the bounding boxes should be
                rescaled to fit the original scale of the image. Defaults to
                False.

        Returns:
            :obj:`InstanceData`: Tracking results of the input images.
            Each InstanceData usually contains ``bboxes``, ``labels``,
            ``scores`` and ``instances_id``.
        """

        if not hasattr(self, 'kf'):
            self.kf = model.motion

        scores = bboxes[:, -1]
        bboxes = bboxes[:, :4]
        valid_inds = scores > self.obj_score_thr
        bboxes = bboxes[valid_inds]
        labels = labels[valid_inds]
        scores = scores[valid_inds]
        embeds = embeds[valid_inds]

        if self.empty or bboxes.size(0) == 0:
            num_new_tracks = bboxes.size(0)
            ids = torch.arange(
                self.num_tracks,
                self.num_tracks + num_new_tracks,
                dtype=torch.long).to(bboxes.device)
            self.num_tracks += num_new_tracks
        else:
            ids = torch.full((bboxes.size(0), ), -1,
                             dtype=torch.long).to(bboxes.device)

            self.tracks, motion_dists = model.motion.track(
                self.tracks, bbox_xyxy_to_cxcyah(bboxes))

            active_ids = self.confirmed_ids

            # reid
            if len(active_ids) > 0:
                track_embeds = self.get(
                    'embeds',
                    active_ids,
                    behavior='mean')
                reid_dists = cosine_distance(track_embeds, embeds)
                valid_inds = [list(self.ids).index(_) for _ in active_ids]
                reid_dists[~np.isfinite(motion_dists[
                    valid_inds, :])] = np.nan

                weight_motion = 0.02
                match_dists = (1 - weight_motion) * reid_dists + \
                    weight_motion * motion_dists[valid_inds]

                # support multi-class association
                track_labels = torch.tensor([
                    self.tracks[id]['labels'][-1] for id in active_ids
                ]).to(bboxes.device)
                cate_match = labels[None, :] == track_labels[:, None]
                cate_cost = ((1 - cate_match.int()) * 1e6).cpu().numpy()
                match_dists = match_dists + cate_cost

                row, col = linear_sum_assignment(match_dists)
                for r, c in zip(row, col):
                    dist = match_dists[r, c]
                    if not np.isfinite(dist):
                        continue
                    if dist <= self.match_score_thr:
                        ids[c] = active_ids[r]

            active_ids = [
                id for id in self.ids if id not in ids
                and self.tracks[id].frame_ids[-1] == frame_id - 1
            ]
            if len(active_ids) > 0:
                active_dets = torch.nonzero(ids == -1).squeeze(1)
                track_bboxes = self.get('bboxes', active_ids)
                ious = bbox_overlaps(track_bboxes, bboxes[active_dets])

                # support multi-class association
                track_labels = torch.tensor([
                    self.tracks[id]['labels'][-1] for id in active_ids
                ]).to(bboxes.device)
                cate_match = labels[None, active_dets] == track_labels[:, None]
                cate_cost = (1 - cate_match.int()) * 1e6

                dists = (1 - ious + cate_cost).cpu().numpy()

                row, col = linear_sum_assignment(dists)
                for r, c in zip(row, col):
                    dist = dists[r, c]
                    if dist < 1 - self.match_iou_thr:
                        ids[active_dets[c]] = active_ids[r]

            new_track_inds = ids == -1
            ids[new_track_inds] = torch.arange(
                self.num_tracks,
                self.num_tracks + new_track_inds.sum(),
                dtype=torch.long).to(bboxes.device)
            self.num_tracks += new_track_inds.sum()

        self.update(
            ids=ids,
            bboxes=bboxes[:, :4],
            scores=scores,
            labels=labels,
            embeds=embeds,
            frame_ids=frame_id)

        return bboxes, labels, ids