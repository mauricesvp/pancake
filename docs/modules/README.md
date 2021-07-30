# Modules

<img width="800" height="%" src="/gitimg/app_structure.png">

# Data + Preprocessing
Pancake offers various different data sources, both live and offline,
as well as multiple sources at once (e.g. multi-camera setups).
The preprocessing necessary is handled "under the hood" (for details, see [datasets.py](pancake/utils/datasets.py) ),
usually the user shouldn't have to delve into this too much though.

For details on how to specify different sources (+ Examples) see [Configurations](https://github.com/mauricesvp/pancake/blob/main/README.md#configurations).

# Backend

Because the Detection can - depending on the data - not necessarily be run directly,
the Backend is responsible for adjusting the data as necessary to make sure the results are in order.
All backends are initialized with an instance of a Detector, which is used for the detection.

| Backend       | Details   | Configuration ```NAME:```         |
| ------------- | -------   | ------------------- |
| Basic         | foo       | ```"simple"```
| DEI           | foo       | ```"dei"```

<details>
  <summary><b>Adding a new Backend</b></summary>
  <ol>
    <li>Create your backend_foo.py within <code>detector/backends/</code> .</li>
    <li>Create a Backend class that inherits from the <a href="pancake/detector/backends/backend.py">Base Backend</a>.</li>
    <li>Implement the <code>detect</code> method.</li>
    <li>Add your Backend to the <a href="pancake/detector/backends/__init__.py">registry</a> (i.e. add <code>from .backend_foo import Foo</code>).</li>
    <li>Set your Backend in the configuration (under "BACKEND" -> NAME: "foo").</li>
  </ol>
Important: When implementing your Backend, you need to stick to the <a href=https://mauricesvp.github.io/pancake/pancake/detector/backends/backend.html> Backend API</a>!
</details>

# Detection

The Detection itself is handled by an instance of a Detector.
For pancake, we provide two versions of the [YOLOv5](https://github.com/ultralytics/yolov5) detector.

| Detection     | Details   | Configuration ```NAME:```         |
| ------------- | -------   | ------------------- |
| YOLOv5 - Simple         | foo       | ```"yolo_simple"```
| YOLOv5 - Custom           | foo       | ```"yolo_custom"```

<details>
  <summary><b>Simple</b></summary>
  A very simple detector using a pretrained model provided by YOLOv5.
  
  **Configuration options:** (under <code>YOLO_SIMPLE:</code>)
  
  | Parameter               | Example Values   | Description         |
  | ---------------------   | ----------------- | ------------------- |
  | <code>size</code> | "s", "m", "l" or "x"         | Yolo model size

</details>

<details>
  <summary><b>Custom</b></summary>
  A detector based on YOLOv5, custom trained with data provided by a previous project group.

  **Configuration options:** (under <code>YOLO_CUSTOM:</code>)


  | Parameter               | Example Values   | Description         |
  | ---------------------   | ----------------- | ------------------- |
  | <code>model</code>                    | "yolov5"          | Yolo custom class from registry
  | <code>weights</code>      | "yolov5m.pt"          | Weights to be loaded
  | <code>img_size</code>      | 640         | Image size, applies to standard and trt
  | <code>conf_thres</code>      | 0.65         | Confidence threshold (Confidence will be in the range 0-1.0)
  | <code>iou_thres</code>      | 0.6         | IoU (Intersection over Union) threshold
  | <code>classes</code>      | [0, 1, 2, 3, 5, 7]         | Filtered classes: Person(0), Bicycle(1), Car(2), Motorcycle(3), Bus(5), Truck(7)
  | <code>agnostic_nms</code>      | True, False        | Agnostic nms (Non-maximum suppression)
  | <code>max_det</code>      | 20        | Maximum detections per infered frame
  | <code>trt</code>      | True, False        | Enable trt engine for inference
  | <code>trt_engine_path</code>      | "yolov5s6.engine"        | Path to locally compiled engine
  | <code>trt_plugin_library</code>      | "libmyplugins.so"        | Path to locally compiled lib

</details>

<details>
  <summary><b>Adding a new Detector</b></summary>
  <ol>
    <li>Create your detector_foo.py within <code>detector/</code> .</li>
    <li>Create a Detector class that inherits from the <a href="pancake/detector/detector.py">Base Detector</a>.</li>
    <li>Implement the <code>detect</code> method.</li>
    <li>Add your Detector to the <a href="pancake/detector/__init__.py">registry</a> (i.e. add <code>from .detector_foo import Foo</code>).</li>
    <li>Set your Detector in the configuration (under "DETECTOR" -> NAME: "foo").</li>
  </ol>
Important: When implementing your Detector, you need to stick to the <a href=https://mauricesvp.github.io/pancake/pancake/detector/detector.html> Detector API</a>!
</details>

# Tracking

Tracking is handled by an instance of a tracker.
For pancake, we provide a Centroid Tracker as well as a DeepSORT Tracker, 
whereat the Trackers have been loosely taken over from previous project groups.

| Tracker       | Details   | Configuration ```NAME:```         |
| ------------- | -------   | ------------------- |
| Centroid         | foo       | ```"centroid"```
| DeepSORT           | foo       | ```"deepsort"```

<details>
  <summary><b>Centroid Tracker</b></summary>
    
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

</details>
<details>
  <summary><b>DeepSORT</b></summary>

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
  
</details>
<details>
  <summary><b>Adding a new Tracker</b></summary>
  <ol>
    <li>Create your tracker_foo.py within <code>tracker/</code> .</li>
    <li>Create a Tracker class that inherits from the <a href="pancake/tracker/tracker.py">Base Tracker</a>.</li>
    <li>Implement the <code>update</code> method.</li>
    <li>Add your Tracker to the <a href="pancake/tracker/__init__.py">registry</a> (i.e. add <code>from .tracker_foo import Foo</code>).</li>
    <li>Set your Tracker in the configuration (under "TRACKER" -> NAME: "foo").</li>
  </ol>
Important: When implementing your Tracker, you need to stick to the <a href=https://mauricesvp.github.io/pancake/pancake/tracker/tracker.html> Tracker API</a>!
</details>
