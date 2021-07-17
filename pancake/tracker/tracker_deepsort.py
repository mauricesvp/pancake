""" DeepSort Interface Class """

from typing import Tuple

import os
import numpy as np
import torch

from .tracker import BaseTracker
from .deep_sort.deep_sort import DeepSort


class DEEPSORT(BaseTracker):
    def __init__(self, cfg: dict, *args, **kwargs):
        """DeepSort Interface Class

        Args:
            cfg (dict): Dictionary containing configurations
        """
        assert (
            "device" in kwargs
        ), "Used device type needs to be specified (cpu, gpu:0, gpu:1)!"

        # We need to jump through some hoops here to allow relative paths
        path = os.path.join(os.path.dirname(__file__), "..", cfg.DEEPSORT.REID_CKPT)
        path_resolved = os.path.abspath(path)
        device = kwargs["device"]
        self.DS = DeepSort(
            path_resolved,
            device=device,
            max_dist=cfg.DEEPSORT.MAX_DIST,
            min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=cfg.DEEPSORT.MAX_AGE,
            n_init=cfg.DEEPSORT.N_INIT,
            nn_budget=cfg.DEEPSORT.NN_BUDGET,
            max_id=cfg.DEEPSORT.MAX_ID,
            use_cuda=True if device != "CPU" else False,
        )

    def update(self, det: torch.Tensor, img: np.ndarray) -> np.ndarray:
        """Transforms and parses detection matrices to the DeepSort update function.

        Args:
            det (torch.Tensor): Detections on (,6) tensor [xyxy, conf, cls]
            img (np.ndarray): Image in BGR [c, w, h]

        Returns:
            np.ndarray: Tracked entities in [x1, y1, x2, y2, centre x, centre y, id, cls id]
        """
        bbox_xywh, confidences, cls = DEEPSORT.transform_detections(det)
        return self.DS.update(bbox_xywh, confidences, img, cls)

    @staticmethod
    def transform_detections(
        det: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform detection vector to numpy.

        Args:
            det (torch.Tensor): Detections on (,6) tensor [xyxy, conf, cls]

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - x1, y1, x2, y2
                - class confidences
                - model-specific class indices
        """
        t_det = det.cpu().detach().numpy()
        return t_det[:, :4], t_det[..., 4], t_det[..., 5]
