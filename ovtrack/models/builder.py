from mmcv.cnn import build_model_from_cfg as build
from mmcv.utils import Registry

MODELS = Registry("model")
TRACKERS = Registry("tracker")
MOTIONS = Registry("motion")

def build_tracker(cfg):
    """Build tracker."""
    return build(cfg, TRACKERS)

def build_motion(cfg):
    """Build tracker."""
    return build(cfg, MOTIONS)

def build_model(cfg, train_cfg=None, test_cfg=None):
    """Build model."""
    return build(cfg, MODELS, dict(train_cfg=train_cfg, test_cfg=test_cfg))