# Modules

<img width="800" height="%" src="/gitimg/app_structure.png">

## Data + Preprocessing
Pancake offers various different data sources, both live and offline,
as well as multiple sources at once (e.g. multi-camera setups).
The preprocessing necessary is handled "under the hood" (for details, see [datasets.py](pancake/utils/datasets.py) ),
usually the user shouldn't have to delve into this too much though.

For details on how to specify different sources (+ Examples) see [Configurations](https://github.com/mauricesvp/pancake/blob/main/README.md#configurations).

## Backend

Because the Detection can - depending on the data - not necessarily be run directly,
the Backend is responsible for adjusting the data as necessary to make sure the results are in order.
All backends are initialized with an instance of a Detector, which is used for the detection.

| Backend       | Details   | Configuration ```NAME:```         |
| ------------- | -------   | ------------------- |
| Basic         | [Details](https://github.com/mauricesvp/pancake/blob/main/docs/modules/backends.md#basic)  | ```"simple"```
| DEI           | [Details](https://github.com/mauricesvp/pancake/blob/main/docs/modules/backends.md#dei-divide-and-conquer)  | ```"dei"```

### Adding a new Backend
  <ol>
    <li>Create your backend_foo.py within <code>detector/backends/</code> .</li>
    <li>Create a Backend class that inherits from the <a href="pancake/detector/backends/backend.py">Base Backend</a>.</li>
    <li>Implement the <code>detect</code> method.</li>
    <li>Add your Backend to the <a href="pancake/detector/backends/__init__.py">registry</a> (i.e. add <code>from .backend_foo import Foo</code>).</li>
    <li>Set your Backend in the configuration (under "BACKEND" -> NAME: "foo").</li>
  </ol>
  Important: When implementing your Backend, you need to stick to the <a href=https://mauricesvp.github.io/pancake/pancake/detector/backends/backend.html> Backend API</a>!

## Detection

The Detection itself is handled by an instance of a Detector.
For pancake, we provide two versions of the [YOLOv5](https://github.com/ultralytics/yolov5) detector.

| Detection     | Details   | Configuration ```NAME:```         |
| ------------- | -------   | ------------------- |
| YOLOv5 - Simple         | [Details](https://github.com/mauricesvp/pancake/blob/main/docs/modules/detector.md#simple)       | ```"yolo_simple"```
| YOLOv5 - Custom           | [Details](https://github.com/mauricesvp/pancake/blob/main/docs/modules/detector.md#custom)       | ```"yolo_custom"```

### Adding a new Detector
  <ol>
    <li>Create your detector_foo.py within <code>detector/</code> .</li>
    <li>Create a Detector class that inherits from the <a href="pancake/detector/detector.py">Base Detector</a>.</li>
    <li>Implement the <code>detect</code> method.</li>
    <li>Add your Detector to the <a href="pancake/detector/__init__.py">registry</a> (i.e. add <code>from .detector_foo import Foo</code>).</li>
    <li>Set your Detector in the configuration (under "DETECTOR" -> NAME: "foo").</li>
  </ol>
Important: When implementing your Detector, you need to stick to the <a href=https://mauricesvp.github.io/pancake/pancake/detector/detector.html> Detector API</a>!

## Tracking

Tracking is handled by an instance of a tracker.
For pancake, we provide a Centroid Tracker as well as a DeepSORT Tracker, 
whereat the Trackers have been loosely taken over from previous project groups.

| Tracker       | Details   | Configuration ```NAME:```         |
| ------------- | -------   | ------------------- |
| Centroid         | [Details](https://github.com/mauricesvp/pancake/blob/main/docs/modules/tracker.md#centroid)       | ```"centroid"```
| DeepSORT           | [Details](https://github.com/mauricesvp/pancake/blob/main/docs/modules/tracker.md#deepsort)       | ```"deepsort"```

### Adding a new Tracker
  <ol>
    <li>Create your tracker_foo.py within <code>tracker/</code> .</li>
    <li>Create a Tracker class that inherits from the <a href="pancake/tracker/tracker.py">Base Tracker</a>.</li>
    <li>Implement the <code>update</code> method.</li>
    <li>Add your Tracker to the <a href="pancake/tracker/__init__.py">registry</a> (i.e. add <code>from .tracker_foo import Foo</code>).</li>
    <li>Set your Tracker in the configuration (under "TRACKER" -> NAME: "foo").</li>
  </ol>
Important: When implementing your Tracker, you need to stick to the <a href=https://mauricesvp.github.io/pancake/pancake/tracker/tracker.html> Tracker API</a>!

## Storage
  The collected data can optionally be stored in a SQLite database (this can enabled in the [configuration](#configurations)).

## Result Processing
  If you are not only interested in the raw results data, but also in visualizations of detections or tracks,
  you can enable and configure this in the [configuration](#https://github.com/mauricesvp/pancake/blob/main/README.md#configurations).

## Analysis
  Pancake currently doesn't offer further analysis on the collected data.
  This is something that could be tackled in the future.
