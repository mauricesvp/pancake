from .tracker import BaseTracker
from .deep_sort.deep_sort import DeepSort

class DEEPSORT(BaseTracker):
    def __init__(self, cfg: dict, *args, **kwargs):
        assert ('device' in kwargs
            ), "Used device type needs to be specified (cpu, gpu:0, gpu:1)!"
        self.tracker = DeepSort(
            cfg.DEEPSORT.REID_CKPT, 
            max_dist=cfg.DEEPSORT.MAX_DIST, 
            min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE, 
            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, 
            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE, 
            max_age=cfg.DEEPSORT.MAX_AGE, 
            n_init=cfg.DEEPSORT.N_INIT, 
            nn_budget=cfg.DEEPSORT.NN_BUDGET, 
            use_cuda=True if kwargs['device'] != 'cpu' else False
        )

    def update(self, det):
        pass

    def get_tracker_flag(self):
        return self.tracker.tracker.flag

    @staticmethod
    def transform_detections(det):
        pass

