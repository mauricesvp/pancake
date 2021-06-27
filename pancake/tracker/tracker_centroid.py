"""Tracker based on Centroid tracking."""
from .tracker import BaseTracker
from typing import Type, Union

from pancake.logger import setup_logger

from collections import OrderedDict
import numpy as np
import torch
import time
import math

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from scipy.spatial import distance as dist
from operator import itemgetter


class CentroidTracker(BaseTracker):
    def __init__(self, cfg: dict, *args, **kwargs) -> None:
        self.l = setup_logger(__name__)  # You can use any name
        self.l.debug("INIT CENTROID TRACKER - Final")
        # constants
        self.MAX_ID = cfg.CENTROID.MAX_ID
        self.MAX_DISAPPEARED = cfg.CENTROID.MAX_DISAPPEARED
        # car match tolerances
        self.DISTANCE_TOLERANCE = cfg.CENTROID.DISTANCE_TOLERANCE
        self.VERTICAL_TOLERANCE = cfg.CENTROID.VERTICAL_TOLERANCE
        # image boundaries regions
        self.FRAME_WIDTH = cfg.CENTROID.FRAME_WIDTH
        self.FRAME_CHANGE_LC = self.FRAME_WIDTH // 3
        self.FRAME_CHANGE_CR = self.FRAME_CHANGE_LC * 2
        # image transition region size
        self.TRANSITION_WIDTH = cfg.CENTROID.TRANSITION_WIDTH
        # lane separators
        self.LANE_SEPARATOR_LL = cfg.CENTROID.LANE_SEPARATOR_LL
        self.LANE_SEPARATOR_LC = cfg.CENTROID.LANE_SEPARATOR_LC
        self.LANE_SEPARATOR_CR = cfg.CENTROID.LANE_SEPARATOR_CR
        self.LANE_SEPARATOR_RR = cfg.CENTROID.LANE_SEPARATOR_RR
        # deregistration zone boundaries
        self.DEREG_ZONE_L = cfg.CENTROID.DEREG_ZONE_L
        self.DEREG_ZONE_R = cfg.CENTROID.DEREG_ZONE_R
        # registration zone boundaries
        self.REG_ZONE_L = cfg.CENTROID.REG_ZONE_L
        self.REG_ZONE_R = cfg.CENTROID.REG_ZONE_R
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        # dicts for return types
        self.bbox = OrderedDict()
        self.confidence = OrderedDict()
        self.classIds = OrderedDict()
        # known previous distant travelled
        self.lastDistTrav = OrderedDict()

    def _centroid(self, vertices):
        x_list = [vertex for vertex in vertices[::2]]
        y_list = [vertex for vertex in vertices[1::2]]
        x = int(sum(x_list) // len(x_list))
        y = int(sum(y_list) // len(y_list))
        return x, y

    def _register(self, centroid, bbox, conf, cls):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        # dont add lastDistTrav
        # add other information
        self.bbox[self.nextObjectID] = bbox
        self.confidence[self.nextObjectID] = conf
        self.classIds[self.nextObjectID] = cls
        # increment objectID
        self.nextObjectID += 1
        if self.nextObjectID >= self.MAX_ID:
            self.nextObjectID = 0

    def _deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # all of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.bbox[objectID]
        del self.confidence[objectID]
        del self.classIds[objectID]
        try:
            del self.lastDistTrav[objectID]
        except Exception:
            pass

    def _return(self):
        outputs = []
        for id in list(self.objects.keys()):
            outputs.append(
                np.array(
                    [
                        self.bbox[id][0],
                        self.bbox[id][1],
                        self.bbox[id][2],
                        self.bbox[id][3],
                        self.objects[id][0],
                        self.objects[id][1],
                        id,
                        self.classIds[id]
                    ],
                    dtype=np.int
                )
            )
        # output np array
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    def _computeLastDist(self, prevPos, currPos):
        return currPos[0] - prevPos[0], currPos[1] - prevPos[1]

    def _isAboveLaneSeparator(self, centroid):
        if centroid[0] < self.FRAME_CHANGE_LC:
            # left image
            lanesepvec = (
                self.FRAME_CHANGE_LC - 0,
                self.LANE_SEPARATOR_LC - self.LANE_SEPARATOR_LL,
            )
            centrvec = (
                self.FRAME_CHANGE_LC - centroid[0],
                self.LANE_SEPARATOR_LC - centroid[1],
            )
        elif centroid[0] > self.FRAME_CHANGE_CR:
            # right image
            lanesepvec = (
                self.FRAME_WIDTH - self.FRAME_CHANGE_CR,
                self.LANE_SEPARATOR_RR - self.LANE_SEPARATOR_CR,
            )
            centrvec = (
                self.FRAME_WIDTH - centroid[0],
                self.LANE_SEPARATOR_RR - centroid[1],
            )
        else:
            # center image
            lanesepvec = (
                self.FRAME_CHANGE_CR - self.FRAME_CHANGE_LC,
                self.LANE_SEPARATOR_CR - self.LANE_SEPARATOR_LC,
            )
            centrvec = (
                self.FRAME_CHANGE_CR - centroid[0],
                self.LANE_SEPARATOR_CR - centroid[1],
            )
        # calculate cross product in 2d
        cross_product = lanesepvec[0] * centrvec[1] - lanesepvec[1] * centrvec[0]
        return cross_product > 0

    def _isInsideDeregistrationZone(self, centroid):
        if centroid[0] < self.DEREG_ZONE_L:
            # left dereg
            return self._isAboveLaneSeparator(centroid)
        elif centroid[0] > self.DEREG_ZONE_R:
            # right dereg
            return not self._isAboveLaneSeparator(centroid)
        else:
            # between deregs
            return False

    def _isInsideRegistrationZone(self, centroid):
        if centroid[0] < self.REG_ZONE_L:
            # left reg
            return not self._isAboveLaneSeparator(centroid)
        elif centroid[0] > self.REG_ZONE_R:
            # right reg
            return self._isAboveLaneSeparator(centroid)
        else:
            # between regs
            return False

    def _isInsideTransitionZone(self, centroid):
        if (
            centroid[0] < self.FRAME_CHANGE_LC + self.TRANSITION_WIDTH
            and centroid[0] > self.FRAME_CHANGE_LC - self.TRANSITION_WIDTH
        ):
            # left transition region
            return True
        elif (
            centroid[0] < self.FRAME_CHANGE_CR + self.TRANSITION_WIDTH
            and centroid[0] > self.FRAME_CHANGE_CR - self.TRANSITION_WIDTH
        ):
            # right transition region
            return True
        else:
            # not in transition regions
            return False

    def _getDist(self, pt1, pt2):
        return math.dist(pt1, pt2)

    def _getLen(self, tup):
        return (tup[0] ** 2 + tup[1] ** 2) ** 0.5

    def _continueMovement(self, objectID):
        objPos = self.objects[objectID]

        try:
            distance = self.lastDistTrav[objectID]
        except Exception:  # object was only seen one frame
            distance = (0, 0)

        if self._isInsideTransitionZone(objPos):
            if objPos[0] < self.FRAME_CHANGE_LC:
                # follow left vector
                dirVect = (
                    0 - self.FRAME_CHANGE_LC,
                    self.LANE_SEPARATOR_LL - self.LANE_SEPARATOR_LC,
                )
            elif objPos[0] > self.FRAME_CHANGE_CR:
                # follow right vector
                dirVect = (
                    self.FRAME_CHANGE_CR - self.FRAME_WIDTH,
                    self.LANE_SEPARATOR_CR - self.LANE_SEPARATOR_RR,
                )
            else:
                # follow center vector
                dirVect = (
                    self.FRAME_CHANGE_LC - self.FRAME_CHANGE_CR,
                    self.LANE_SEPARATOR_LC - self.LANE_SEPARATOR_CR,
                )

            # flip direction of dirVect for bottom lane
            if not self._isAboveLaneSeparator(objPos):
                dirVect = (-dirVect[0], -dirVect[1])

            # change distance according to dirVect
            lenDistance = self._getLen(distance)
            dirVectLen = self._getLen(dirVect)
            dirVectNorm = (dirVect[0] / dirVectLen, dirVect[1] / dirVectLen)
            distance = (dirVectNorm[0] * lenDistance, dirVectNorm[1] * lenDistance)

        predmove = objPos[0] + distance[0], objPos[1] + distance[1]
        return predmove

    # TODO: make it dependent on the screen and adjust it to be more infront of the car
    def _isDistanceInsideLimit(self, currdst, objectID, matchPos):
        if (currdst < self.DISTANCE_TOLERANCE) and (
            abs(self.objects[objectID][1] - matchPos[1]) < self.VERTICAL_TOLERANCE
        ):
            return True
        return False
        # predpt = self._continueMovement(objectID)

        # currpt = self.objects[objectID]
        # preddst = self._getDist(matchPos, predpt)

        # inRect = self._isInsideRect(currpt, predpt, matchPos)

        # if currdst < self.DISTANCE_TOLERANCE or \
        #     preddst < self.DISTANCE_TOLERANCE or \
        #     inRect:
        #     return True

        # return False

    ###################################
    ## UPDATE
    ###################################

    def update(
        self, det: Type[torch.Tensor], img: Type[np.ndarray]
    ) -> np.ndarray:  # det: list of koordinates x,y , x,y, ...
        # keep track of time for debugging
        update_time_start = int(round(time.time() * 1000))
        # get inputs
        bbox_xyxy, conf, cls = self.transform_detections(det)

        # filter none car and truck objects from input
        # bbox_xyxy_tmp = []
        # conf_tmp = []
        # cls_tmp = []
        # for i, c in enumerate(cls):
        #     if (c < 4 or c == 5 or c == 7):
        #         bbox_xyxy_tmp.append(bbox_xyxy[i])
        #         conf_tmp.append(conf[i])
        #         cls_tmp.append(cls[i])
        # bbox_xyxy = bbox_xyxy_tmp
        # conf = conf_tmp
        # cls = cls_tmp

        # check if no detections are present
        if len(bbox_xyxy) == 0:
            # loop over any existing tracked objects and mark them as disappeared
            for objectID in list(self.disappeared.keys()):
                # increase disappeared count
                self.disappeared[objectID] += 1
                # continue movement
                self.objects[objectID] = self._continueMovement(objectID)
                # deregister if lifetime is surpassed or object drove away
                if (self.disappeared[objectID] > self.MAX_DISAPPEARED or
                    self._isInsideDeregistrationZone(
                        self.objects[objectID]
                )):
                    self._deregister(objectID)

            # return early
            return self._return()

        # initialize an array for the inputs
        inputCentroids = np.zeros((len(bbox_xyxy), 2), dtype="int")
        inputBBOX = np.zeros((len(bbox_xyxy), 4), dtype="int")
        inputConfidence = np.zeros((len(bbox_xyxy), 1), dtype="float")
        inputClass = np.zeros((len(bbox_xyxy), 1), dtype="int")
        # loop over the bounding box rectangles
        for (i, bb) in enumerate(bbox_xyxy):
            # use the bounding box coordinates to derive the centroid
            inputCentroids[i] = self._centroid(bb)
            inputBBOX[i] = bb
            inputConfidence[i] = conf[i]
            inputClass[i] = cls[i]

        # if currently no objects are being tracked
        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                # only register when not in deregistration zone
                if not self._isInsideDeregistrationZone(inputCentroids[i]):
                    self._register(
                        inputCentroids[i],
                        inputBBOX[i],
                        inputConfidence[i],
                        inputClass[i],
                    )

        # otherwise match existing to detected centroids
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            # compute distances
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            # create sorted list of tuples of indexes and value in ascending order
            D_sorted = sorted(np.ndenumerate(D), key=itemgetter(1))
            # keep track of used Rows and Cols
            usedRows = set()
            usedCols = set()

            # loop through row,col and distance
            for line in D_sorted:
                row = line[0][0]
                col = line[0][1]
                distance = line[1]
                # if row or col already used, skip
                if row in usedRows or col in usedCols:
                    continue
                # get objectID
                objectID = objectIDs[row]
                # otherwise a match is created but only if inside a maximum distance
                if self._isDistanceInsideLimit(distance, objectID, inputCentroids[col]):
                    # safe the distance travelled
                    self.lastDistTrav[objectID] = self._computeLastDist(
                        self.objects[objectID], inputCentroids[col]
                    )
                    # update object
                    self.objects[objectID] = inputCentroids[col]
                    self.bbox[objectID] = inputBBOX[col]
                    self.confidence[objectID] = inputConfidence[col]
                    self.classIds[objectID] = inputClass[col]
                    self.disappeared[objectID] = 0
                    # remove Row and Col
                    usedRows.add(row)
                    usedCols.add(col)

            # get not matched rows and cols
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # unmatched objects will be counted as disappeared
            for row in unusedRows:
                # get object ID
                objectID = objectIDs[row]
                # increase disappeared count
                self.disappeared[objectID] += 1
                # continue movement
                self.objects[objectID] = self._continueMovement(objectID)
                # deregister if lifetime is surpassed or object drove away
                if (self.disappeared[objectID] > self.MAX_DISAPPEARED or 
                    self._isInsideDeregistrationZone(
                        objectCentroids[row]
                )):
                    self._deregister(objectID)

            # unmatched input centroids will get registered
            for col in unusedCols:
                if not self._isInsideDeregistrationZone(inputCentroids[col]):
                    if self._isInsideRegistrationZone(inputCentroids[col]):
                        self._register(
                            inputCentroids[col],
                            inputBBOX[col],
                            inputConfidence[col],
                            inputClass[col],
                        )

        # keep track of time for debugging
        update_time_end = int(round(time.time() * 1000))
        self.l.debug(
            "Centroid update took {} ms".format(update_time_end - update_time_start)
        )

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
