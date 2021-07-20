"""
Simple Backend
----------------
TODOs:
    * Apply ROIs
    * Filter by object class id
      (this could be done in the detector as well)

"""
from typing import List, Tuple, Type

import torch
import cv2
import numpy as np

from .backend import Backend
from ..detector import Detector


class SIMPLE(Backend):
    def __init__(
        self, detector: Type[Detector], roi: List[int] = None, *args, **kwargs
    ) -> None:
        """[summary]

        Args:
            detector (Type[Detector]): [description]
            roi (List[int], optional): [description]. Defaults to None.
        """
        """

        :param detector: Detector which provides 'detect' method,
                         which can take one or multiple images.

        """
        self.detector = detector
        if roi:
            self.roi = [v for v in roi.values()][:-2]
        else:
            self.roi = None

    def detect(
        self, source: List[np.ndarray]
    ) -> Tuple[List[torch.Tensor], List[np.ndarray]]:
        """[summary]

        Args:
            source (List[np.ndarray]): [description]

        Returns:
            Tuple[List[torch.Tensor], List[np.ndarray]]: [description]
        """
        """Detect objects on image(s).

        :param source: Image or list of images.
        """
        if self.roi:
            assert len(self.roi) == len(source)

            cropped = []
            offsets = []
            for i, img in enumerate(source):
                x0, y0, x1, y1 = self.roi[i]
                offsets.append((x0, y0))
                cropped.append(img[y0:y1, x0:x1])

            dets = self.detector.detect(cropped)

            res = torch.Tensor(dets[0])
            for i, det in enumerate(dets):
                for x in det:
                    tlx, tly = offsets[i]
                    h, w, _ = source[i].shape
                    x[0] += w * (i) + tlx
                    x[1] += tly
                    x[2] += w * (i) + tlx
                    x[3] += tly
                res = torch.cat((res, det), dim=0)
        else:
            dets = self.detector.detect(source)
            res = torch.Tensor(dets[0])
            for i, det in enumerate(dets[1:]):
                for x in det:
                    h, w, _ = source[i + 1].shape
                    x[0] += w * (i + 1)
                    x[2] += w * (i + 1)
                res = torch.cat((res, det), dim=0)
        if not type(source) is list:
            source = [source]
        frame = cv2.hconcat([*source])
        return res, frame
