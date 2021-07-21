""" DeepSort Wrapper Class """
import numpy as np
import torch

""" DeepSort Wrapper Class """

from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker


__all__ = ["DeepSort"]


class DeepSort(object):
    def __init__(
        self,
        model_path: str,
        device: str = "CPU",
        max_dist: float = 0.2,
        min_confidence: float = 0.3,
        nms_max_overlap: float = 1.0,
        max_iou_distance: float = 0.7,
        max_age: int = 70,
        n_init: int = 3,
        nn_budget: int = 100,
        max_id: int = 100,
        use_cuda: bool = True,
    ):
        """ DeepSort wrapper class

        Args:
            model_path (str): Feature extractor model path
            device (str, optional): Device to leverage. Defaults to "CPU".
            max_dist (float, optional): Max cosine distance to nearest neighbor. Defaults to 0.2.
            min_confidence (float, optional): Min confidence to be considered. Defaults to 0.3.
            nms_max_overlap (float, optional): Non-maximum Suppression max overlap fraction. \
                                                ROIs that overlap more than this values are suppressed. \
                                                Defaults to 1.0.
            max_iou_distance (float, optional): Max intersection over union distance. 
                                                Associations with cost larger than this value are disregarded. \
                                                Defaults to 0.7.
            max_age (int, optional): Maximum number of missed timesteps before a track is deleted. Defaults to 70.
            n_init (int, optional): Number of consecutive detections before the track is confirmed. The
                                    track state is set to `Deleted` if a miss occurs within the first
                                    `n_init` frames. Defaults to 3.
            nn_budget (int, optional): fix samples per class to at most this number. Removes the oldest samples \
                                        when the budget is reached. Defaults to 100.
            max_id (int, optional): Highest possible id for tracked entity. Defaults to 100.
            use_cuda (bool, optional): Fallback to CUDA device if possible. Defaults to True.
        """
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap

        self.extractor = Extractor(model_path, device=device, use_cuda=use_cuda)

        max_cosine_distance = max_dist
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(
            metric,
            max_iou_distance=max_iou_distance,
            max_age=max_age,
            n_init=n_init,
            max_id=max_id,
        )

    def update(
        self,
        bbox_xyxy: np.ndarray,
        confidences: np.ndarray,
        ori_img: np.ndarray,
        cls: np.ndarray,
    ) -> np.ndarray:
        """Updates the internal state.

        Description:
            - Runs NMS
            - Gets the extracted features
            - Updates the tracker state
            - Outputs the track identities

        Args:
            bbox_xyxy (np.ndarray): Detected bounding boxes in [x1, y1, x2, y2]
            confidences (np.ndarray): Confidence scores of the detected entities
            ori_img (np.ndarray): Original image in BGR [c, w, h]
            cls (np.ndarray): Model-specific class indices of the detected entities

        Returns:
            np.ndarray: Tracked entities in [x1, y1, x2, y2, centre x, centre y, id, cls id]
        """
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        bbox_xyxy = np.asarray(bbox_xyxy, dtype=int)
        features = self._get_features(bbox_xyxy, ori_img)
        bbox_tlwh = np.asarray([self._xyxy_to_tlwh(xyxy) for xyxy in bbox_xyxy])
        detections = [
            Detection(bbox_tlwh[i], conf, features[i], cls[i])
            for i, conf in enumerate(confidences)
            if conf > self.min_confidence
        ]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            track_posx = track.pos_x
            track_posy = track.pos_y
            cls = track.cls
            outputs.append(
                np.array(
                    [x1, y1, x2, y2, track_posx, track_posy, track_id, cls],
                    dtype=np.int,
                )
            )
        if outputs:
            outputs = np.stack(outputs, axis=0)
        return outputs

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.0
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.0
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        return t, l, int(x2 - t), int(y2 - l)

    def _get_features(self, bbox_xyxy, ori_img):
        im_crops = []
        for box in bbox_xyxy:
            x1, y1, x2, y2 = box
            im = ori_img[y1:y2, x1:x2]
            if im.any():
                im_crops.append(im)
        return self.extractor(im_crops) if im_crops else np.array([])
