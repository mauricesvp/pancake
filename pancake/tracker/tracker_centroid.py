"""Tracker based on Centroid tracking."""
from .tracker import BaseTracker
from typing import Type, Union

import numpy as np
import torch


class CentroidTracker(BaseTracker):
    def __init__(self, cfg: dict, *args, **kwargs) -> None:
        print("INIT CENTROID TRACKER")

    def _centroid(self, vertices):
        x_list = [vertex for vertex in vertices[::2]]
        y_list = [vertex for vertex in vertices[1::2]]
        x = sum(x_list) // len(x_list)
        y = sum(y_list) // len(y_list)
        return x, y
    
    def update(self, det: Type[torch.Tensor], img: Type[np.ndarray]) -> np.ndarray:  # det: list of koordinates x,y , x,y, ...
        print("UPDATE CENTROID TRACKER")
        bbox_xywh, confidences, _ = self.transform_detections(det)
        centroids = [[5,6,7,8, 1,2, 0] for b in bbox_xywh]

        print(centroids)
        #return centroids
        
        outputs = []

        outputs.append(
                np.array(
                    [5,6,7,8, 1,2, 0], dtype=np.int
                )
            )

        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    @staticmethod
    def transform_detections(det: Type[torch.Tensor]):
        """
        Transform detection vector to numpy

        :param det (torch.Tensor): prediction tensor

        :return xyxy (np.ndarray (,4)): x1, y1, x2, y2
                conf (np.ndarray (,1)): class confidences
                cls  (np,ndarray (,1)): class indeces
        """
        t_det = det.cpu().detach().numpy()
        return t_det[:, :4], t_det[..., 4], t_det[..., 5]