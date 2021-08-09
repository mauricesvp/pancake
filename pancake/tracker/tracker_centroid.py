"""Tracker based on Centroid tracking."""
from .tracker import BaseTracker
from typing import Type, Tuple

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
        """Initializes the CentroidTracker

        Description:
            Sets up the internal variables and allocates the OrderedDicts for further processing.

        Args:
            cfg (dict): dictionary including all configuration variables.
        """        
        self.l = setup_logger(__name__)  # You can use any name
        self.l.debug("INIT CENTROID TRACKER - Final")
        # options
        self.USE_BETTER_RECTS = cfg.CENTROID.USE_BETTER_RECTS
        self.USE_DYNAMIC_SCALING = cfg.CENTROID.USE_DYNAMIC_SCALING
        # constants
        self.MAX_ID = cfg.CENTROID.MAX_ID
        self.MAX_DISAPPEARED = cfg.CENTROID.MAX_DISAPPEARED
        # car match tolerances
        self.DISTANCE_TOLERANCE = cfg.CENTROID.DISTANCE_TOLERANCE
        self.VERTICAL_TOLERANCE = cfg.CENTROID.VERTICAL_TOLERANCE
        self.FRONT_DISTANCE_TOLERANCE = cfg.CENTROID.FRONT_DISTANCE_TOLERANCE
        self.BACK_DISTANCE_TOLERANCE = cfg.CENTROID.BACK_DISTANCE_TOLERANCE
        self.SIDE_DISTANCE_TOLERANCE = cfg.CENTROID.SIDE_DISTANCE_TOLERANCE
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

    def _centroid(self, vertices: np.ndarray) -> Tuple[int, int]:
        """Calculates a centroid from the BBOX coordinates

        Description:
            May receive an unlimited amount of coordinates. But these coordinates need to alternate between x and y.

        Args:
            vertices (np.ndarray): array of x and y coordinates

        Returns:
            Tuple[int, int]: tuple describing the x and y coordinates on the image
        """        
        x_list = [vertex for vertex in vertices[::2]]
        y_list = [vertex for vertex in vertices[1::2]]
        x = int(sum(x_list) // len(x_list))
        y = int(sum(y_list) // len(y_list))
        return x, y

    def _register(self, centroid: np.ndarray, bbox: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        """Registers an object

        Args:
            centroid (np.ndarray): 2 element array of the coordinates of the object
            bbox (np.ndarray): 4 element array of the top left and bottom right bounding box limits of the object
            conf (np.ndarray): 1 element array of the confidence of the object
            cls (np.ndarray): 1 element array of the class of the object
        """        
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

    def _deregister(self, objectID: int):
        """Deletes an object by ID

        Args:
            objectID (int): ID of the object to be deleted
        """        
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

    def _return(self) -> np.ndarray:
        """Returns the Centroids with their respective data as required by the update function.

        Returns:
            np.ndarray: Tracked entities in [x1, y1, x2, y2, centre x, centre y, id, cls id]
        """        
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

    def _computeLastDist(self, prevPos: np.ndarray, currPos: np.ndarray) -> Tuple[int,int]:
        """Computes the distance vector between two points

        Description:
            Used as a measure between the last known position and the predicted next position

        Args:
            prevPos (np.ndarray): 2 element array of the x and y coordinates in the image of the previously known position (first position)
            currPos (np.ndarray): 2 element array of the x and y coordinates in the image of the currently known position (second position)

        Returns:
            Tuple[int,int]: 2 element tuple of the x and y coordinates in the image of the vector from the previous to the current position
        """        
        return currPos[0] - prevPos[0], currPos[1] - prevPos[1]

    def _isAboveLaneSeparator(self, centroid: np.ndarray) -> bool:
        """Checks if a centroid is on the upper or lower lane on the image

        Args:
            centroid (np.ndarray): 2 element array of the x and y coordinates on the image of the checked position

        Returns:
            bool: Returns if the input centroid is on the upper (True) or lower (False) driving lane
        """        
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

    def _isInsideDeregistrationZone(self, centroid: np.ndarray) -> bool:
        """Determines whether a centroid is inside the deregistration zone on the left and right cameras edges

        Args:
            centroid (np.ndarray): 2 element array of the x and y coordinates on the image of the position being checked

        Returns:
            bool: Returns whether the input centroid is inside the deregistration zone
        """
        if centroid[0] < self.DEREG_ZONE_L:
            # left dereg
            return self._isAboveLaneSeparator(centroid)
        elif centroid[0] > self.DEREG_ZONE_R:
            # right dereg
            return not self._isAboveLaneSeparator(centroid)
        else:
            # between deregs
            return False

    def _isInsideRegistrationZone(self, centroid: np. ndarray) -> bool:
        """Determines whether a centroid is inside the registration zone on the left and right cameras edges

        Args:
            centroid (np.ndarray): 2 element array of the x and y coordinates on the image of the position being checked

        Returns:
            bool: Returns whether the input centroid is inside the registration zone
        """
        if centroid[0] < self.REG_ZONE_L:
            # left reg
            return not self._isAboveLaneSeparator(centroid)
        elif centroid[0] > self.REG_ZONE_R:
            # right reg
            return self._isAboveLaneSeparator(centroid)
        else:
            # between regs
            return False

    def _isInsideTransitionZone(self, centroid: np.ndarray) -> bool:
        """Determines whether a centroid is inside the transition region between camera images

        Args:
            centroid (np.ndarray): 2 element array of the x and y coordinates on the image of the position being checked

        Returns:
            bool: Returns whether the input centroid is inside the transition region
        """        
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

    def _getLen(self, tup: Tuple[int, int]) -> float:
        """Gets the length of a vector depicted as a tuple

        Args:
            tup (Tuple[int, int]): 2 element tuple describing a vector

        Returns:
            float: length of the vector
        """        
        return (tup[0] ** 2 + tup[1] ** 2) ** 0.5

    def _continueMovement(self, objectID: int) -> Tuple[int, int]:
        """Continues the movement of a specific object.

        Description:
            Moves an object - depicted by its objectID - to a position where it is predicted to appear in the future.
            This is done by using the last know movement.

        Args:
            objectID (int): ID of the object that should be moved.

        Returns:
            Tuple[int, int]: 2 element tuple of the new predicted position
        """        
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

    def _isInsideRect(self, currpt: np.ndarray, predpt: Tuple[int, int], matchpt: np.ndarray, objID: int) -> bool:
        """Checks if matchpt is inside a rectangle

        Description:
            The rectangle is generated by the current position (currpt) and the predicted position (predpt) with some dynamically or statically adjustable padding

        Args:
            currpt (np.ndarray): 2 element array of the x and y coordinates on the image of the current position
            predpt (Tuple[int, int]): 2 element tuple of the x and y coordinates on the image of the predicted position
            matchpt (np.ndarray): 2 element array of the x and y coordinates on the image of the potential match
            objID (int): ID of the object currently inspected

        Returns:
            bool: Returns if matchpt is inside a rectangle
        """        
        if self.USE_DYNAMIC_SCALING:
            # get scaling based on camera
            if currpt[0] < self.FRAME_CHANGE_LC:
                # left camera
                objectSize = 0.0863384 * currpt[0] - 109.93
            elif currpt[0] > self.FRAME_CHANGE_CR:
                # right camera
                objectSize = -0.060652 * currpt[0] + 676.94
            else:
                # center camera
                objectSize = 228
            # normalize scale to center camera
            scale = objectSize / 228
        else:
            scale = 1

        # set scaling
        vectorScale = scale
        paddingScale = scale

        # get vector from currpt to predpt
        predVect = (predpt[0] - currpt[0], predpt[1] - currpt[1])

        # set own prediction vector if no prediction was made previously
        if predVect == (0,0):
            if currpt[0] < self.FRAME_CHANGE_LC:
                # follow left vector
                predVectTmp = (
                    0 - self.FRAME_CHANGE_LC,
                    self.LANE_SEPARATOR_LL - self.LANE_SEPARATOR_LC,
                )
            elif currpt[0] > self.FRAME_CHANGE_CR:
                # follow right vector
                predVectTmp = (
                    self.FRAME_CHANGE_CR - self.FRAME_WIDTH,
                    self.LANE_SEPARATOR_CR - self.LANE_SEPARATOR_RR,
                )
            else:
                # follow center vector
                predVectTmp = (
                    self.FRAME_CHANGE_LC - self.FRAME_CHANGE_CR,
                    self.LANE_SEPARATOR_LC - self.LANE_SEPARATOR_CR,
                )

            # flip direction of predVectTmp for bottom lane
            if not self._isAboveLaneSeparator(currpt):
                predVectTmp = (-predVectTmp[0], -predVectTmp[1])

            # normalize predVectTmp
            predVectLen = self._getLen(predVectTmp)
            predVectNorm = (predVectTmp[0] / predVectLen, predVectTmp[1] / predVectLen)

            # generate predVect
            predVect = (predVectNorm[0] * vectorScale, predVectNorm[1] * vectorScale)

        # calculate prediction vectors to generate the rectangle from
        predVectLen    = self._getLen(predVect)
        predVectNorm   = (predVect[0] / predVectLen, predVect[1] / predVectLen)
        predVect90Norm = (-predVectNorm[1], predVectNorm[0])

        # calculate padding vectors
        frontPadding  = (self.FRONT_DISTANCE_TOLERANCE * paddingScale *  predVectNorm[0],
                         self.FRONT_DISTANCE_TOLERANCE * paddingScale *  predVectNorm[1])
        backPadding   = (self.BACK_DISTANCE_TOLERANCE  * paddingScale * -predVectNorm[0],
                         self.BACK_DISTANCE_TOLERANCE  * paddingScale * -predVectNorm[1])
        leftPadding   = (self.SIDE_DISTANCE_TOLERANCE  * paddingScale *  predVect90Norm[0],
                         self.SIDE_DISTANCE_TOLERANCE  * paddingScale *  predVect90Norm[1])
        rightPadding  = (self.SIDE_DISTANCE_TOLERANCE  * paddingScale * -predVect90Norm[0],
                         self.SIDE_DISTANCE_TOLERANCE  * paddingScale * -predVect90Norm[1])

        # generate rectangle points
        frpt = (predpt[0] + frontPadding[0] + rightPadding[0],
                predpt[1] + frontPadding[1] + rightPadding[1])
        flpt = (predpt[0] + frontPadding[0] + leftPadding[0],
                predpt[1] + frontPadding[1] + leftPadding[1])
        brpt = (currpt[0] + backPadding[0]  + rightPadding[0],
                currpt[1] + backPadding[1]  + rightPadding[1])
        blpt = (currpt[0] + backPadding[0]  + leftPadding[0],
                currpt[1] + backPadding[1]  + leftPadding[1])
        
        # create objects
        point = Point(matchpt[0], matchpt[1])
        polygon = Polygon([flpt, frpt, brpt, blpt])

        # check
        contains = polygon.contains(point)

        return contains

    def _isDistanceInsideLimit(self, currdst: int, objectID: int, matchPos: np.ndarray) -> bool:
        """Checks if a match between currdst and matchPos is possible / inside a specified area

        Description:
            May use a static or dynamically adjustable method to check if the match Position (matchPos) is within a maximum distance to the current position

        Args:
            currdst (int): distance to match position
            objectID (int): ID of the object being checked
            matchPos (np.ndarray): 2 element array of the x and y coordinate on the image of the objects potential position

        Returns:
            bool: Returns if matchPos is a possible match for the object at ID objectID
        """             
        if self.USE_BETTER_RECTS:
            # get objects predicted next position
            predpt = self._continueMovement(objectID)
            # get current objects position
            currpt = self.objects[objectID]
            # check if matchPos is inside the possible movement area
            inRect = self._isInsideRect(currpt, predpt, matchPos, objectID)

            return inRect
        else:
            # check horizontal and vertical distance
            # worse performance for wide camera angles
            if (currdst < self.DISTANCE_TOLERANCE) and (
                abs(self.objects[objectID][1] - matchPos[1]) < self.VERTICAL_TOLERANCE
            ):
                return True
            return False

    ###################################
    ## UPDATE
    ###################################

    def update(
        self, det: Type[torch.Tensor], img: Type[np.ndarray]
    ) -> np.ndarray:  # det: list of koordinates x,y , x,y, ...
        """Updates the internal states of the Centroid tracker.

        Description:
            This function should be called every frame.

            Centroids will be premoved automatically when a detections was not possible.

            Centroids will be removed after a specified amount of frames when no longer detected.


        Args:
            det (torch.Tensor): Detections on (,6) tensor [xyxy, conf, cls]
            img (Type[np.ndarray]): Image in BGR [c, w, h] (not needed for Centroid)

        Returns:
            np.ndarray: Tracked entities in [x1, y1, x2, y2, centre x, centre y, id, cls id]
        """        
        # keep track of time for debugging
        update_time_start = int(round(time.time() * 1000))
        # get inputs
        bbox_xyxy, conf, cls = self.transform_detections(det)

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
        """Transform detection vector to numpy.
        Args:
            det (torch.Tensor): Detections on (,6) tensor [xyxy, conf, cls]
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                 [(x1, y1, x2, y2),
                  (class confidences),
                  (model-specific class indices]
        """
        t_det = det.cpu().detach().numpy()
        return t_det[:, :4], t_det[..., 4], t_det[..., 5]
