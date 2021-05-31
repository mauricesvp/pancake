"""Tracker based on Centroid tracking."""
from .tracker import BaseTracker
from typing import Type, Union

from pancake.logger import setup_logger

from collections import OrderedDict
import numpy as np
import torch

from scipy.spatial import distance as dist


class CentroidTracker(BaseTracker):
    def __init__(self, cfg: dict, *args, **kwargs) -> None:
        self.l = setup_logger(__name__)  # You can use any name
        self.l.debug("INIT CENTROID TRACKER - BASIC")
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        # More dicts for return types
        self.bbox = OrderedDict()
        self.confidence = OrderedDict()
        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = cfg.CENTROID.MAX_DISAPPEARED

    def _centroid(self, vertices):
        x_list = [vertex for vertex in vertices[::2]]
        y_list = [vertex for vertex in vertices[1::2]]
        x = int( sum(x_list) // len(x_list) )
        y = int( sum(y_list) // len(y_list) )
        return x, y

    def _return(self):
        outputs = []
        for id in list(self.objects.keys()):
            outputs.append(
                np.array(
                    [self.bbox[id][0],self.bbox[id][1], self.bbox[id][2],self.bbox[id][3], 
                    self.objects[id][0], self.objects[id][1], id], dtype=np.int
                )
            )

        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        
        self.l.debug(outputs)

        return outputs

    def _register(self, centroid, bbox, conf):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        # add other information
        self.bbox[self.nextObjectID] = bbox
        self.confidence[self.nextObjectID] = conf
        # increment objectID
        self.nextObjectID += 1

    def _deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # all of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.bbox[objectID]
        del self.confidence[objectID]
    
    def update(self, det: Type[torch.Tensor], img: Type[np.ndarray]) -> np.ndarray:  # det: list of koordinates x,y , x,y, ...
        self.l.debug("UPDATE CENTROID TRACKER")
        bbox_xyxy, conf, _ = self.transform_detections(det)
        
        #print("TRANSFORM:")
        #print(bbox_xyxy, conf)

        # check to see if the list of input bounding box rectangles
        # is empty
        if len(bbox_xyxy) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self._deregister(objectID)
            # return early as there are no centroids or tracking info
            # to update
            return self._return()

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(bbox_xyxy), 2), dtype="int")
        inputBBOX = np.zeros((len(bbox_xyxy), 4), dtype="int")
        inputConfidence = np.zeros((len(bbox_xyxy), 1), dtype="float")
        # loop over the bounding box rectangles
        for (i, bb) in enumerate(bbox_xyxy):
            # use the bounding box coordinates to derive the centroid
            inputCentroids[i] = self._centroid(bb)
            inputBBOX[i] = bb
            inputConfidence[i] = conf[i]

        #print("INPUTS:")
        #print(inputCentroids, inputBBOX, inputConfidence)

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self._register(inputCentroids[i], inputBBOX[i], inputConfidence[i])
        
        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()
            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()
            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue
                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.bbox[objectID] = inputBBOX[col]
                self.confidence[objectID] = inputConfidence[col]
                self.disappeared[objectID] = 0
                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self._deregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self._register(inputCentroids[col], inputBBOX[col], inputConfidence[col])
        # return the set of trackable objects
        return self._return()


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