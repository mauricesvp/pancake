import numpy as np
import torch
from typing import Type, Union

from .tracker import BaseTracker
from .deep_sort.deep_sort import DeepSort

class DEEPSORT(BaseTracker):
    def __init__(self, 
                 cfg: dict, 
                 *args, 
                 **kwargs):
        assert ('device' in kwargs
            ), "Used device type needs to be specified (cpu, gpu:0, gpu:1)!"

        self.DS = DeepSort(
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

    def update(self, 
               det: Type[torch.Tensor], 
               img: Type[np.ndarray]
               ) -> np.ndarray:
        """
        Tracker update function

        :param det (np.darray (,6)):  detections array 
        :param img (np.darray (,3)): original image

        :return (np.darray): [x1, y1, x2, y2, centre x, centre y, id]
        """
        bbox_xywh, confidences, _ = DEEPSORT.transform_detections(det)
        return self.DS.update(bbox_xywh, confidences, img)

    def get_tracker_flag(self):
        return self.DS.tracker.flag

    @staticmethod
    def transform_detections(det: Type[torch.Tensor]):
        """
        Transform detection vector to numpy and from xyxy to xywh

        :param det (torch.Tensor): prediction tensor

        :return xywh (np.ndarray (,4)): (center coordinates) x, y, width, height
                conf (np.ndarray (,1)): class confidences
                cls  (np,ndarray (,1)): class indeces
        """
        t_det = det.cpu().detach().numpy()
        xywh, conf, cls = DEEPSORT.xyxy_to_xywh(t_det[:, :4]), t_det[..., 4], t_det[..., 5] 

        return xywh, conf, cls

    @staticmethod
    def xyxy_to_xywh(boxes_xyxy: Union[np.ndarray, torch.Tensor]):
        """
        Helper function to transform array containing data in xyxy to xywh

        :param boxes_xyxy (np.ndarray (n,4)): array containing n xyxy box coordinates
        :return boxes_xywh (np.ndarray (n,4))
        """
        if isinstance(boxes_xyxy, torch.Tensor):
            boxes_xywh = boxes_xyxy.clone()
        elif isinstance(boxes_xyxy, np.ndarray):
            boxes_xywh = boxes_xyxy.copy()

        boxes_xywh[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.
        boxes_xywh[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.
        boxes_xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        boxes_xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

        return boxes_xywh