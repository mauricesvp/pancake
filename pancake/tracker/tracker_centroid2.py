"""Tracker based on Centroid tracking."""
from .tracker import BaseTracker

import time
from operator import itemgetter
from scipy.spatial import distance as dist #normal
from collections import OrderedDict #normal
import numpy as np #normal

#import config as cfg
#import database as db



def centroid(vertices):
    x_list = [vertex for vertex in vertices[::2]]
    y_list = [vertex for vertex in vertices[1::2]]
    x = sum(x_list) // len(x_list)
    y = sum(y_list) // len(y_list)
    return (x, y)


class CentroidTracker(BaseTracker):
    def __init__(self, cfg: dict, *args, **kwargs) -> None: #normal
        print("INIT CENTROID TRACKER")
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.previousPos = OrderedDict()
        self.nextObjectID = 0 #normal
        self.objects = OrderedDict() #normal
        self.disappeared = OrderedDict() #normal
        self.continued_movement = OrderedDict()
        self.lastValidMovement = OrderedDict()
        self.length = OrderedDict()
        self.height = OrderedDict()
        self.conf = OrderedDict()
        self.class_id = OrderedDict()
        self.DBList = []

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.MAX_DISAPPEARED = cfg.MAX_DISAPPEARED #normal
        self.previous_timestamp = None
        self.time_between_frames = 1
        
        #copy cfg
        self.cfg = cfg

    def register(self, centroid, bbox, conf, class_id):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid #normal
        self.previousPos[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0 #normal
        self.continued_movement[self.nextObjectID] = False
        self.lastValidMovement[self.nextObjectID] = None
        self.length[self.nextObjectID] = bbox[0]
        self.height[self.nextObjectID] = bbox[1]
        self.conf[self.nextObjectID] = conf
        self.class_id[self.nextObjectID] = class_id
        self.nextObjectID += 1 #normal

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID] #normal
        del self.previousPos[objectID]
        del self.disappeared[objectID] #normal
        del self.continued_movement[objectID]
        del self.lastValidMovement[objectID]
        del self.length[objectID]
        del self.height[objectID]
        del self.conf[objectID]
        del self.class_id[objectID]

    def continueMovement(self, objectID, verticalTolerance):
        # check if values for the last valid movement is existing (i.e. the object must have been detected at least once in two consecutive frames)
        if self.lastValidMovement[objectID] is not None:
            self.objects[objectID][0] = self.objects[objectID][0] + \
                                        (self.lastValidMovement[objectID][0] * self.time_between_frames)
            self.continued_movement[objectID] = True

    def addToDatabase(self, frame_timestamp, frame_date, frame_time, objectID):
        if self.cfg.SKIP_DB:
            return
        object_for_db = (frame_timestamp,
                         frame_date,
                         frame_time,
                         objectID,
                         int(self.objects[objectID][0]),
                         int(self.objects[objectID][1]),
                         int(self.length[objectID]),
                         int(self.height[objectID]),
                         int(self.class_id[objectID]),
                         self.conf[objectID],
                         self.continued_movement[objectID],
                         int(round(time.time() * 1000)))
        self.DBList.append(object_for_db)

    def pushToDatabase(self, conn):
        if len(self.DBList) == 0 or self.cfg.SKIP_DB:
            return

        #db.insert_detections(conn, self.DBList)
        self.DBList = []

    def update(self, det):  # det: list of koordinates x,y , x,y, ...
        centroids = [centroid(d) for d in det]
        return centroids

    def update(self, det, confidences, class_ids):
        # create current timestamp in ms
        frame_timestamp = int(round(time.time() * 1000))
        frame_date = time.strftime('%Y%m%d')
        frame_time = time.strftime('%H%M')

        if self.previous_timestamp is not None and not self.cfg.IS_VIDEO_INPUT:
            self.time_between_frames = frame_timestamp - self.previous_timestamp

        # create db connection
        #conn = db.create_connection(self.cfg.DATABASE_PATH)

        # check to see if the list of input bounding box rectangles
        # is empty
        if len(det) == 0: #normal
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()): #normal
                self.continued_movement[objectID] = False
                self.disappeared[objectID] += 1 #normal
                self.continueMovement(objectID, self.cfg.VERTICAL_TOLERANCE)
                self.previousPos[objectID] = None
                # if we have reached a maximum of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                # first two ifs check if vehicle is on edge of lane and deregister object immediately if so
                if (self.objects[objectID][1] < self.cfg.LANE_SEPARATOR and self.objects[objectID][0] < self.cfg.DEREGISTRATION_ZONE) \
                        or (self.objects[objectID][1] > self.cfg.LANE_SEPARATOR \
                            and self.objects[objectID][0] > (self.cfg.FRAME_WIDTH-self.cfg.LANE_SEPARATOR)) \
                        or (self.disappeared[objectID] > self.MAX_DISAPPEARED): #normal
                    self.deregister(objectID) #normal

            # add each object to database
            for objectID in self.objects.keys():
                self.addToDatabase(frame_timestamp, frame_date, frame_time, objectID)
            # return early as there are no centroids or tracking info
            # to update
            #self.pushToDatabase(conn)
            self.previous_timestamp = frame_timestamp
            return self.objects

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(det), 2), dtype="int") #normal
        inputBBoxes = np.zeros((len(det), 2), dtype="int")
        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(det): #normal
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0) #normal
            cY = int((startY + endY) / 2.0) #normal
            inputCentroids[i] = (cX, cY) #normal
            inputBBoxes[i] = (endX - startX, endY - startY)
        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0: #normal
            for i in range(0, len(inputCentroids)): #normal
                # check if vehicle is not on edge of lane
                if (inputCentroids[i][1] < self.cfg.LANE_SEPARATOR and inputCentroids[i][0] >= self.cfg.DEREGISTRATION_ZONE) \
                        or (inputCentroids[i][1] > self.cfg.LANE_SEPARATOR \
                            and inputCentroids[i][0] <= (self.cfg.FRAME_WIDTH-self.cfg.DEREGISTRATION_ZONE)) \
                        or self.cfg.IGNORE_REGISTRATION_ZONES:
                    self.register(inputCentroids[i], inputBBoxes[i], confidences[i], class_ids[i]) #normal

                # otherwise, are are currently tracking objects so we need to
                # try to match the input centroids to existing object centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys()) #normal
            objectCentroids = list(self.objects.values()) #normal
            # compute the distance between each pair of object centroids and input centroids
            D = dist.cdist(np.array(objectCentroids), inputCentroids) #normal
            # create sorted list of tuples of indexes and value in ascending order
            D_sorted = sorted(np.ndenumerate(D), key=itemgetter(1))
            usedRows = set() #normal
            usedCols = set() #normal
            # loop through pairs of object, inputCentroid, and distance
            for x in D_sorted:
                row = x[0][0]
                col = x[0][1]
                distance = x[1]
                if row in usedRows or col in usedCols: #normal
                    continue #normal
                else:
                    if (distance < self.cfg.DISTANCE_TOLERANCE) and \
                            (abs(self.objects[objectIDs[row]][1] - inputCentroids[col][1]) < self.cfg.VERTICAL_TOLERANCE):
                        objectID = objectIDs[row] #normal
                        self.objects[objectID] = inputCentroids[col] #normal
                        self.length[objectID] = inputBBoxes[col][0]
                        self.height[objectID] = inputBBoxes[col][1]
                        self.conf[objectID] = confidences[col]
                        self.class_id[objectID] = class_ids[col]
                        self.disappeared[objectID] = 0 #normal
                        self.continued_movement[objectID] = False
                        # save the movement between the last two frames (only, if it is "forward")
                        if self.previousPos[objectID] is not None:
                            if (self.objects[objectID][1] < 302 \
                                    and (self.objects[objectID][0] - self.previousPos[objectID][0]) < 0) \
                                or (self.objects[objectID][1] > 302 \
                                    and (self.objects[objectID][0] - self.previousPos[objectID][0]) > 0):
                                self.lastValidMovement[objectID] = (self.objects[objectID] - self.previousPos[
                                    objectID]) / self.time_between_frames
                            self.previousPos[objectID] = self.objects[objectID]
                        # indicate that we have examined each of the row and column indexes, respectively
                        usedRows.add(row) #normal
                        usedCols.add(col) #normal
            # compute both the row and column index we have NOT yet examined
            # object ids that could not be assigned
            unusedRows = set(range(0, D.shape[0])).difference(usedRows) #normal
            # input centroids that could not be assigned
            unusedCols = set(range(0, D.shape[1])).difference(usedCols) #normal

            # loop over the unused row indexes
            for row in unusedRows:
                # grab the object ID for the corresponding row
                # index and increment the disappeared counter
                objectID = objectIDs[row]
                self.continued_movement[objectID] = False
                self.disappeared[objectID] += 1
                self.continueMovement(objectID, self.cfg.VERTICAL_TOLERANCE)
                self.previousPos[objectID] = None
                # check to see if the number of consecutive
                # frames the object has been marked "disappeared"
                # for warrants deregistering the object
                # check if vehicle was on edges of the lanes or has exceeded max_disappeared
                if (objectCentroids[row][1] < self.cfg.LANE_SEPARATOR and objectCentroids[row][0] < self.cfg.DEREGISTRATION_ZONE) \
                        or (objectCentroids[row][1] > self.cfg.LANE_SEPARATOR \
                            and objectCentroids[row][0] > (self.cfg.FRAME_WIDTH-self.cfg.DEREGISTRATION_ZONE)) \
                        or (self.disappeared[objectID] > self.MAX_DISAPPEARED):
                    self.deregister(objectID)

                    # usedRows.add(row)

            # register each unused inputCentroid as a new object:
            for col in unusedCols:
                if (inputCentroids[col][1] < self.cfg.LANE_SEPARATOR and inputCentroids[col][0] >= self.cfg.DEREGISTRATION_ZONE) \
                        or (inputCentroids[col][1] > self.cfg.LANE_SEPARATOR \
                            and inputCentroids[col][0] <= (self.cfg.FRAME_WIDTH-self.cfg.DEREGISTRATION_ZONE)) \
                        or self.cfg.IGNORE_REGISTRATION_ZONES: 
                    self.register(inputCentroids[col], inputBBoxes[col], confidences[col], class_ids[col])

        # add each object to database
        for objectID in self.objects.keys():
            self.addToDatabase(frame_timestamp, frame_date, frame_time, objectID)
        # return the set of trackable objects        
        #self.pushToDatabase(conn)
        self.previous_timestamp = frame_timestamp
        return self.objects
