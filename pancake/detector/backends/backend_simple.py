"""
Simple Backend
----------------
TODOs:
    * Apply ROIs
    * Filter by object class id
      (this could be done in the detector as well)

"""
import torch

from .backend import Backend


class SIMPLE(Backend):
    def __init__(self, detector, *args, **kwargs) -> None:
        """

        :param detector: Detector which provides 'detect' method,
                         which can take one or multiple images.

        """
        self.detector = detector

    def detect(self, source) -> list:
        """Detect objects on image(s).

        :param source: Image or list of images.
        """
        # TODO: Apply ROIs
        dets = self.detector.detect(source)
        res = torch.Tensor(dets[0])
        for i, det in enumerate(dets[1:]):
            for x in det:
                h, w, _ = source[i + 1].shape
                x[0] += w * (i + 1)
                x[2] += w * (i + 1)
            res = torch.cat((res, det), dim=0)
        return res
