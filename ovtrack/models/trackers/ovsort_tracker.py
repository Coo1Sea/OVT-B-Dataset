import cv2
import mmcv
import numpy as np
import os
import random
import seaborn as sns
import torch
import torch.nn.functional as F
from collections import defaultdict
from mmcv.image import imread, imwrite
from mmcv.visualization import color_val, imshow
from mmdet.core import bbox_overlaps
from motmetrics.lap import linear_sum_assignment
from ovtrack.core.bbox import bbox_xyxy_to_cxcyah, bbox_cxcyah_to_xyxy
from ovtrack.core import cal_similarity
from ..builder import TRACKERS
from .sort_tracker import SortTracker

@TRACKERS.register_module()
class OVSortTracker(SortTracker):
    def __init__(
        self,
        obj_score_thr=0.0001,
        match_score_thr=0.5,
        num_frames_retain=10,
        momentum_embed=0.8,
        motion_weight=0.02,
        init_cfg=None,
        **kwargs
    ):
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.obj_score_thr = obj_score_thr
        self.match_score_thr = match_score_thr
        self.num_frames_retain = num_frames_retain
        self.momentums = {'embeds': momentum_embed}
        self.motion_weight = motion_weight
        self.reset()

    def update_track(self, id, obj):
        super().update_track(id, obj)
        for k, v in zip(self.memo_items, obj):
            v = v[None]
            if self.momentums is not None and k in self.momentums:
                m = self.momentums[k]
                self.tracks[id][k] = (1 - m) * self.tracks[id][k] + m * v
            else:
                self.tracks[id][k].append(v)

        bbox = bbox_xyxy_to_cxcyah(self.tracks[id].bboxes[-1])  # size = (1, 4)
        assert bbox.ndim == 2 and bbox.shape[0] == 1
        bbox = bbox.squeeze(0).cpu().numpy()
        self.tracks[id].mean, self.tracks[id].covariance = self.kf.update(
            self.tracks[id].mean, self.tracks[id].covariance, bbox)

    def track(
        self,
        model,
        bboxes,
        labels,
        embeds,
        cls_embeds,
        frame_id,
        **kwargs
    ):

        if not hasattr(self, 'kf'):
            self.kf = model.motion

        if embeds is None:
            ids = torch.full((bboxes.size(0),), -1, dtype=torch.long)
            return bboxes, labels, ids

        bboxes, labels, embeds = self.remove_distractor(bboxes, labels, track_feats=embeds)

        if bboxes.size(0) > 0 and not self.empty:

            ids = torch.full((bboxes.size(0),), -1, dtype=torch.long)
            active_ids = [id for id, _ in self.tracks.items()]

            pred_bbox = torch.zeros((len(self.tracks), 4), device=bboxes.device)
            for i, (track_id, track) in enumerate(self.tracks.items()):
                predict_mean, _ = model.motion.predict(track['mean'], track['covariance'])
                pred_bbox[i] = torch.tensor(predict_mean[:4], dtype=torch.float32, device=bboxes.device)

            pred_bbox_xyxy = bbox_cxcyah_to_xyxy(pred_bbox)
            iou_dists = bbox_overlaps(bboxes[:,:4], pred_bbox_xyxy)
            
            track_embeds = self.get('embeds', active_ids)
            sims = torch.mm(embeds, track_embeds.t())
            exps = torch.exp(sims)
            d2t_scores = exps / (exps.sum(dim=1).view(-1, 1) + 1e-6)
            t2d_scores = exps / (exps.sum(dim=0).view(1, -1) + 1e-6)
            scores = (d2t_scores + t2d_scores) / 2

            cos_embeds = F.normalize(embeds, p=2, dim=1)
            cos_track_embeds = F.normalize(track_embeds, p=2, dim=1)
            cos = torch.mm(cos_embeds, cos_track_embeds.t())
            cos = (1.0 + cos) / 2
            reid_dists = (scores + cos) / 2
            
            match_dists = (1 - self.motion_weight) * reid_dists + self.motion_weight * iou_dists
            match_dists = 1.0 - match_dists.T
            row, col = linear_sum_assignment(match_dists.cpu().numpy())
            for r, c in zip(row, col):
                dist = match_dists[r, c]
                if dist <= self.match_score_thr:
                    ids[c] = active_ids[r]

            ids = ids.to(bboxes.device) 
            new_track_inds = ids == -1
            ids[new_track_inds] = torch.arange(
                self.num_tracks,
                self.num_tracks + new_track_inds.sum(),
                dtype=torch.long).to(bboxes.device)
            self.num_tracks += new_track_inds.sum()

        else:
            ids = torch.full((bboxes.size(0),), -1, dtype=torch.long)
            num_new_tracks = bboxes.size(0)
            ids = torch.arange(
                self.num_tracks,
                self.num_tracks + num_new_tracks,
                dtype=torch.long).to(bboxes.device)
            self.num_tracks += num_new_tracks

        self.update(
            ids=ids, 
            bboxes=bboxes[:, :4], 
            scores=bboxes[:, -1], 
            labels=labels, 
            embeds=embeds, 
            frame_ids=frame_id)
        return bboxes, labels, ids
    
    def remove_distractor(
        self,
        bboxes,
        labels,
        track_feats,
        object_score_thr=0.5,
        distractor_nms_thr=0.3,

    ):
        valid_inds = labels > -1
        # nms
        low_inds = torch.nonzero(
            bboxes[:, -1] < object_score_thr, as_tuple=False
        ).squeeze(1)
        
        ious = bbox_overlaps(bboxes[low_inds, :-1], bboxes[:, :-1])
        for i, ind in enumerate(low_inds):
            if (ious[i, :ind] > distractor_nms_thr).any():
                valid_inds[ind] = False

        bboxes = bboxes[valid_inds]
        labels = labels[valid_inds]
        embeds = track_feats[valid_inds]

        return bboxes, labels, embeds