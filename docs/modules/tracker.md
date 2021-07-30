# Tracker

## Centroid    
  **Configuration options:** (under <code>CENTROID:</code>)
  
  | Parameter               | Example Values   | Description         |
  | ---------------------   | ----------------- | ------------------- |
  | <code>TRACKER_CFG_PATH</code> | "../configs/tracker/centroid.yaml"       | Centroid config path
  
      
  **centroid.yaml:**
  
  | Parameter               | Example Values   | Description         |
  | ---------------------   | ----------------- | ------------------- |
  | <code>MAX_ID</code> | 10000       | Limit for the Track IDs
  | <code>MAX_DISAPPEARED</code> | 10       | Maximum time (in frames) an object will be premoved on disappearance
  | <code>DISTANCE_TOLERANCE</code> | 500       | Maximum distance to allow for a car tracking match
  | <code>VERTICAL_TOLERANCE</code> | 100       | Maximum vertical distance to allow for a car tracking match
  | <code>FRAME_WIDTH</code> | 11520       | Total image width
  | <code>TRANSITION_WIDTH</code> | 200       | Transition width around the image edges
  | <code>LANE_SEPARATOR_LL</code> | 1117       | y-coordinates of the separator line - left
  | <code>LANE_SEPARATOR_LC</code> | 925       | y-coordinates of the separator line - left-center
  | <code>LANE_SEPARATOR_CR</code> | 925       | y-coordinates of the separator line - right-center
  | <code>LANE_SEPARATOR_CR</code> | 1151       | y-coordinates of the separator line - right
  | <code>DEREG_ZONE_L</code> | 1600       | Deregistration zone x-boundary left
  | <code>DEREG_ZONE_R</code> | 10500       | Deregistration zone x-boundary right
  | <code>REG_ZONE_L</code> | 2750       | Registration zone x-boundary left
  | <code>REG_ZONE_R</code> | 9750       | Registration zone x-boundary right

## DeepSORT
  **Configuration options:** (under <code>DEEPSORT:</code>)
  
  | Parameter               | Example Values   | Description         |
  | ---------------------   | ----------------- | ------------------- |
  | <code>TRACKER_CFG_PATH</code> | "../configs/tracker/deep_sort.yaml"       | DeepSORT config path
  
      
  **deep_sort.yaml:**
  
  | Parameter               | Example Values   | Description         |
  | ---------------------   | ----------------- | ------------------- |
  | <code>REID_CKPT</code> | "../weights/tracker/deepsort/feature_extractor.t7"       | -
  | <code>MAX_DIST</code> | 0.6       | -
  | <code>MIN_CONFIDENCE</code> | 0.4       | -
  | <code>NMS_MAX_OVERLAP</code> | 0.7       | -
  | <code>MAX_IOU_DISTANCE</code> | 0.75       | -
  | <code>MAX_AGE</code> | 70       | -
  | <code>N_INIT</code> | 3       | -
  | <code>NN_BUDGET</code> | 10000       | -
  | <code>MAX_ID</code> | 100000       | -
