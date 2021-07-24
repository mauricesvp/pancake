# pancake :pancakes: - Panorama Camera Car Tracking

[<img src="https://user-images.githubusercontent.com/26842284/125350856-a71c9a80-e35f-11eb-9302-b9e648d37e6e.gif" width="90%">](https://user-images.githubusercontent.com/26842284/125350856-a71c9a80-e35f-11eb-9302-b9e648d37e6e.gif)

_pancake_ is an application for panorama camera car tracking. It comes with a simple and modular program design facilitating easy implementation and application of different techniques regarding panorama stitching, object detection and object tracking. 

**Following features are included:**
* Straight forward implementation and application of state-of-the-art panorama stitching, object detection and object tracking technologies
* Include a discretionary number of image streams of various source types
* Several options for result visualization 
* Optional database logging of vehicle tracks with Sqlite3 
* Modular structure for extension of new functionalities and approaches

##### Documentation
[The most recent documentation can be found here](https://mauricesvp.github.io/pancake/pancake/).

<!------------------------- ToC --------------------------->
## Table of Contents
- [pancake :pancakes: - Panorama Camera Car Tracking](#pancake-pancakes---panorama-camera-car-tracking)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Poetry](#poetry)
    - [Pipenv, Virtualenv](#pipenv-virtualenv)
    - [(Optional) Additional Software](#optional-additional-software)
      - [NVIDIA Driver, CUDA Toolkit, cuDNN](#nvidia-driver-cuda-toolkit-cudnn)
      - [OpenCV CUDA](#opencv-cuda)
  - [Usage](#usage)
    - [Quickstart](#quickstart)
    - [Configurations](#configurations)
  - [Modules](#modules)
    - [Backend](#backend)
    - [Detection](#detection)
    - [Tracking](#tracking)
  - [Further Notes](#further-notes)
    - [Google Colab](#google-colab)
    - [TensorRT](#tensorrt)
    - [Samples](#samples)
  - [Used Third Party Repos](#used-third-party-repos)
  - [License](#license)
  - [Authors](#authors)

<!------------------------- Installation --------------------------->
## Installation

### Poetry

>Poetry is arguably Python's most sophisticated dependency management option available today. Poetry goes far beyond dependencies, with features like generating .lock files, generating project scaffolding, and a ton of configuration options, all of which are handled via a simple CLI. If you're unsure how to cleanly and effectively structure and manage your Python projects, do yourself a favor and use Poetry. [Source](https://hackersandslackers.com/python-poetry-package-manager/)

1. Make sure ```Poetry``` and ```Python3.8``` are installed:

  ```bash 
    poetry --version
    which python3.8
  ```
How to install: [```Poetry```](https://python-poetry.org/docs/), [```Python3.8```](https://linuxize.com/post/how-to-install-python-3-8-on-ubuntu-18-04/)

2. Set a target directory for this repo:

  ```bash
    # this sets a temporary system variable, e.g. TARGET_DIR=~/DCAITI
    export TARGET_DIR=*target directory*   
  ```

3. Clone our repo into the desired location: 

  ```bash
    cd $TARGET_DIR
    
    # either via HTTPS
    git clone https://github.com/mauricesvp/pancake.git

    # or via SSH
    git clone git@github.com:mauricesvp/pancake.git
  ```

4. Afterwards, navigate to the pancake location and install the dependencies:
   
  ```bash
    cd $TARGET_DIR/pancake
    poetry install
  ``` 

5. Finally, activate the virtual environment and run the main script:

  ```bash
    poetry shell
    python pancake/run.py
  ```

**For more information on basic Poetry usage refer to: https://python-poetry.org/docs/basic-usage/**


<details>
  <summary><b>Troubleshoot</b></summary>
  <br>

  * When trying to install the dependencies:
>The current project's Python requirement (X.X.XX) is not compatible with some of the required packages Python requirement:

1. Navigate to pancake directory and delete the ```poetry.lock```:
  ```bash
    cd $TARGET_DIR/pancake
    sudo rm poetry.lock
  ```

2. Then, let poetry know we want to use ```Python3.8```: (find out the location via ```which python3.8```)
  ```bash
    poetry env use *path to python3.8*
  ```

3. Now, try to install the dependencies again:
  ```bash
    poetry install
  ```
</details>

<br>

### Pipenv, Virtualenv

We definitely recommend to use _Poetry_ as python package manager. Still, in case you want to use _Virtualenv_ or _Pipenv_, we provide a ```requirements.txt``` and  ```dev-requirements.txt```.

1. Clone our repo into a desired location: 

  ```bash
    cd $TARGET_DIR
    
    # either via HTTPS
    git clone https://github.com/mauricesvp/pancake.git

    # or via SSH
    git clone git@github.com:mauricesvp/pancake.git
  ```

2. Create a Pipenv or Virtualenv with ```Python3.8```

3. Now, activate your python environment and install the dependencies:
  ```bash 
    source *path to env*/bin/activate     # Pipenv
    # or
    workon *venv name*                    # Virtualenv

    pip install -r requirements.txt       # Base packages
    pip install -r dev-requirements.txt   # Development packages
  ```
4. Have fun cooking up some pancakes:
  ```bash
    python run.py
  ```

<details>
  <summary><b>Troubleshoot</b></summary>
  <br>

</details>

<br>

### (Optional) Additional Software 
A high processing throughput is essential to allow for live tracking with our app. In order to fully leverage local computing capabilities, it is of considerable importance to source the GPU. Our experiments have shown that live application is virtually impossible without considering the latter for computations. Thus, utilizing the below mentioned softwares might be crucial. 

#### NVIDIA Driver, CUDA Toolkit, cuDNN
Our application was tested on CUDA versions **>=10.1**.

[We recommend this tutorial for installation](https://medium.com/@stephengregory_69986/installing-cuda-10-1-on-ubuntu-20-04-e562a5e724a0)

#### OpenCV CUDA
Our application was tested on OpenCV versions **>=4.5**. 

[We recommend this tutorial for installation](https://www.sproutworkshop.com/2021/04/how-to-compile-opencv-4-5-2-with-cuda-11-2-and-cudnn-8-1-on-ubuntu-20-04/)

**Note**: 
* After compilation, validate if OpenCV is able to access your CUDA device:
1. Activate the project specific python environment:
```bash
  cd $TARGET_DIR/pancake
  poetry shell
  python
```
2. Now the python shell will open and you can check, if your CUDA device is available via:
```python
  import cv2
  print(cv2.cuda.getCudaEnabledDeviceCount())
```
* Proceed with removing ```opencv-python``` from the python environment. Otherwise, python will fallback to the CPU version of OpenCV.

<!------------------------- Usage --------------------------->
## Usage

### Quickstart
After you have followed the steps from the [installation](#installation), simply start the main script with:

```bash
  cd $TARGET_DIR/pancake

  poetry shell              # activate the venv
  python pancake/run.py
```

### Configurations
All of the pancake ingredients can simply be specified in the designated _[pancake.yaml](https://github.com/mauricesvp/pancake/blob/ab9f80588563f4d753fb6add980d1b76aaa5b6f6/pancake/pancake.yaml)_. Below, you will find a detailed description on the underlying parameters: 

<!-- Device -->
<details>
  <summary><b>Device</b></summary>
  <br>

  Select a processing device the app should leverage. 

  **Possible values:**
  * ```DEVICE```: _"CPU", "GPU", "0", "1", ..._

  **Note**: _"GPU"_ is the same device as _"0"_
</details>

<!-- Logging -->
<details>
  <summary><b>Logging</b></summary>
  <br>

  Select a level of verbose program output.

  **Possible values:**
  * ```LEVEL```: _"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"_
</details>

<!-- Database -->
<details>
  <summary><b>Database</b></summary>
  <br>

  Specify if the vehicle tracks should be logged in an external database.

  **Possible values:**
  * ```STORE```: _"True", "False"_
  * ```SCHEME PATH```: _path to yaml file containing custom db schema_
  * ```FILENAME```: _name of the stored database file_

  **Note**: 
  * When using a ```SCHEME PATH``` different to the default, it is necessary to adapt _[pancake/pancake/db.py](https://github.com/mauricesvp/pancake/blob/ab9f80588563f4d753fb6add980d1b76aaa5b6f6/pancake/db.py)_. Critical parts of the code are marked as such!
  * If you use the same database file for multiple runs, the database will contain data from respective execution.
  * The [default database design](https://github.com/mauricesvp/pancake/blob/a5b243825c83719a19d286c2a9df2f0bb8bb2132/configs/database/extended_db.yaml) is displayed below:
<!-- <iframe width="560" height="315" src='https://dbdiagram.io/embed/60d8f344dd6a59714821ae88'> </iframe> -->
<img width="400" height="%" src="/gitimg/default_db_schema.png">
</details>

<!-- Data -->
<details>
  <summary><b>Data</b></summary>
  <br>

  Specify a data source/s to retrieve images from as well as a region of interest to be applied on the latter. 

  **SOURCE**

  There are several types of sources one can whip into the pancake dough. Essentially, the quantity of provided sources determine the number of frames to be assembled into a panorama.

  | Source type           | Description                                              | Example | Note |
  | -----------           | -----------                                              | --------  | ---- |
  | **Image**             | Path to single image                                     | ```"../samples/r45/1c/1621796022.9767.jpg"``` | (None) |
  | **Video**             | Path to single video                                     | ```"../samples/output.avi"```| (None) |
  | **Sequence of Images**| Path to directory holding several images                 | ```"../samples/r45/1c"```| (None) |
  | **Directories with Image Sequences/Videos**| (yaml) List of (multiple) directories| <img width="250" height="%" src="/gitimg/source_list.png">  | The directories are only allowed to contain the same type of source (either images or videos)|
  | **Live Streams**      | Path to .txt file containing stream adresses | ```"../samples/streams.txt"``` | Stream adresses could be from an IP camera, YouTube, Twitch and more. [Example content](samples/streams.txt) |

**Note**: For database logging with correct timestamps, it is required that the images are named after their respective timestamp. Livestreams on the other hand are timed by the exact stamp the frame was polled. For videos from the past, there currently is no according timestamp strategy available.

  **ROI**

  Region of interests can be specified by providing the yaml file a dictionary containing the upper left and bottom right *x, y* coordinates of the region for each seperate frame.

  **Example**

  <img width="170" height="%" src="/gitimg/example_ROI.png">

</details> 

<!-- Backend -->
<details>
  <summary><b>Backend</b></summary>
  <br>

  Specify the backend related configurations.

  **Possible values:**
  * ```NAME```: _name of the backend strategy according to the registry_
  * ```DEI```: ```SIMPLE```: _True_, _False_ (enables simpler version of DEI)

  **Note**: For more information on the backend registry and which strategies are currently implemented, refer to [Backend](#backend).

</details>

<!-- Detector -->
<details>
  <summary><b>Detector</b></summary>
  <br>

  Specify the detector related configurations.

  **Possible values:**
  * ```NAME```: _name of the detector technology according to the registry_

  **Note**: For more information on the detector registry and which detector technologies are currently implemented, refer to [Detection](#detection).
</details>

<!-- Tracker -->
<details>
  <summary><b>Tracker</b></summary>
  <br>

  Specify the tracker related configurations.

  **Possible values:**
  * ```NAME```: _name of the tracking algorithm according to the registry_

  **Note**: For more information on the tracker registry and which tracking algorithms are currently implemented, refer to [Tracking](#tracking).
</details>

<!-- Result Processing -->
<details>
  <summary><b>Result Processing</b></summary>
  <br>

  **General**
  | Parameters        | Possible Values   | Description         |
  | ---------------   | ---------------   | ------------------- |
  | ```VIEW_RES```    | _"True", "False"_ | Visualize the most recent (enriched) frame 
  | ```SAVE_RES```    | _"True", "False"_ | Save results (more detailed configurations under _Saving_)
  | ```ASYNC_PROC```  | _"True", "False"_ | Asynchronous result processing (a designated slave process is spawned to postprocess the frames)
  | ```DEBUG```       | _"True", "False"_ | Allows manual frame stepping |

  **Note**: 
  - ```VIEW_RES``` can't be true, when ```ASYNC_PROC``` is turned on (```cv2.imshow``` not callable from within a subprocess)
  - Enabling ```ASYNC_PROC``` yields significant speedup
  - ```DEBUG``` is only available when the processed frame is shown
  
  **Draw Options**

  The parameters below make up the main visualization controllers. (applies when ```VIEW_RES``` or ```SAVE_RES``` is true)
  | Parameters                | Possible Values   | Description         |
  | ---------------------     | ----------------- | ------------------- |
  | ```DRAW_DET```            | _"True", "False"_ | Draw the detection bounding boxes
  | ```DRAW_TRACKS```         | _"True", "False"_ | Draw the tracked bounding boxes
  | ```DRAW_TRACK_HIST```     | _"True", "False"_ | Draw the corresponding tracks to the bounding boxes (draws a line representing the tracked route of the vehicle)
  | ```MAX_TRACK_HIST_LEN```  | Integer           | Max track history length (max number of tracks matrices saved/considered for the track history visualization)

  **Draw Details**

  The parameters below give you more detailed options for visualization. (applies when ```VIEW_RES``` or ```SAVE_RES``` is true)
  | Parameters                | Possible Values   | Description         |
  | ------------------------- | ---------------   | ------------------- |
  | ```HIDE_LABELS```         | _"True", "False"_ | Hide detected class labels and track ids
  | ```HIDE_CONF```           | _"True", "False"_ | Hide the detection confidences
  | ```LINE_THICKNESS```      | Integer           | General line and annotation thickness

  **Asynchronous Queue**

  These configurations concern the queue that is used to store the stitched images, detection matrix and tracks matrix sended from the main process to the designated results-processing subprocess. (applies when ```ASYNC_PROC``` is true)
  | Parameters        | Possible Values   | Description         |
  | ---------------   | ----------------- | ------------------- |
  | ```Q_SIZE```      | Integer           | Queue size
  | ```PUT_BLOCKED``` | _"True", "False"_ | When true, main loop is stopped for ```PUT_TIMEOUT``` seconds until a slot is freed, will otherwise raise an exception
  | ```PUT_TIMEOUT``` | Float             | Max waiting time (in s) for feeding recent data into the queue, will throw exception when time ran out

  **Note**:
  - the queue is filled when result processing is slower than the actual detection and tracking  

  **Saving**

  Below parameters represent granular saving options. (applies when ```SAVE_RES``` is true) 
  | Parameters        | Possible Values     | Description         |
  | ---------------   | -----------------   | ------------------- |
  | ```MODE```        | _"image" or "video"_| Save the resulting frames either as images or a video
  | ```PATH```        | String              | Relative save directory
  | ```SUBDIR```      | String              | Target subdirectory under ```PATH```, will be imcremented automatically after each run
  | ```VID_FPS```     | Integer             | FPS of the resulting video (when ```MODE``` = _"video"_)
  | ```EXIST_OK```    | _"True", "False"_   | Do not increment automatically (keep saving in ```PATH```/```SUBDIR```)

  **Note**:
  - the images and videos are named after the timestamp when the respective frame gets saved
</details>

<br>

[Back to ToC](#table-of-contents)

<!------------------------- Modules --------------------------->
## Modules

<img width="800" height="%" src="/gitimg/app_structure.png">

The pancake framework can be thought of as a data pipeline. The incoming data is preprocessed,
the backend generates detections using a detector, the tracker generates tracks,
and the results are stored in a database (this happens for every frame).

Pancake has been designed with modularity in mind, that is to say the Backend, Detector and Tracker can easily be changed,
which also means new ones can be implemented and integrated easily.

Find more details on how to write your own Backend, Detector or Tracker below.

### Data + Preprocessing
  Pancake offers various different data sources, both live and offline,
  as well as multiple sources at once (e.g. multi-camera setups).
  The preprocessing necessary is handled "under the hood" (for details, see [datasets.py](pancake/utils/datasets.py) ),
  usually the user shouldn't have to delve into this too much.

### Backend
  Because the Detection can - depending on the data - not necessarily be run directly,
  the Backend is responsible for adjusting the data as necessary to make sure the results are in order.
  All backends are initialized with an instance of a Detector, which is used for the detection.
  <br>
  <details>
    <summary><b>Basic</b></summary>
    The Basic Backend simply takes the input image(s), and runs the detection on each image.
  </details>
  <details>
    <summary><b>DEI (Divide and Conquer)</b></summary>
    The DEI Backend is specifically designed for the detection on the Strasse des 17. Juni,
    using a panorama image (made up by three images).
    Because the detections would be very poor if it was run one the panorama directly,
    the Backend first splits the panorama image into partial images.
    These then get rotated, depending on the proximity to the center (no rotation in the center, more rotation on the outer sides).
    This is done as the angle of the cars gets quite skewed on the outer sides, which hinders a successful detection.
    The actual detection is now run on the partial images, after which the rotation und splitting are reversed to produce the final results.
  </details>
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

### Detection
The Detection itself is handled by an instance of a Detector.
For pancake, we provide two versions of the [YOLOv5](https://github.com/ultralytics/yolov5) detector.
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

### Tracking
  Tracking is handled by an instance of a tracker.
  For pancake, we provide a Centroid Tracker as well as a DeepSORT Tracker, 
  whereat the Trackers have been loosely taken over from previous project groups.
  <details>
    <summary><b>Centroid Tracker</b></summary>
  </details>
  <details>
    <summary><b>DeepSORT</b></summary>
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

### Storage
  The collected data can optionally be stored in a SQLite database (this can enabled in the [configuration](#configurations)).

### Result Processing
  If you are not only interested in the raw results data, but also in visualizations of detections or tracks,
  you can enable this in the [configuration](#configurations).

### Analysis
  Pancake currently doesn't offer further analysis on the collected data.
  This is something that could be tackled in the future.

[Back to ToC](#table-of-contents)

<!------------------------- Further Notes --------------------------->
## Further Notes

### Google Colab

[Google Colab for training Yolov5 models on custom data](https://colab.research.google.com/drive/1xtMJhFlp0cB9S2_irIkHAjJ9_6Tol-g9?usp=sharing)

[Google Colab for executing pancake](https://colab.research.google.com/drive/1NGkIHqXnOPeZqV1RbvcGoQ_DTjOgpGxC?usp=sharing)


### TensorRT
[External repo for generating TRT engines](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5)


### Samples
[drive](https://drive.google.com/drive/folders/1Y8FiPtxZiQrv7BrF05uWrDqHdZ8bA11h)

[Back to ToC](#table-of-contents)

## Used Third Party Repos
* **Yolov5**, Ultralytics: https://github.com/ultralytics/yolov5
* **DeepSORT**: https://github.com/nwojke/deep_sort
* **Centroid Tracker**: https://gist.github.com/adioshun/779738c3e28151ffbb9dc7d2b13c2c0a

## License

[<img width='80' src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/MIT_logo.svg/1920px-MIT_logo.svg.png">](https://opensource.org/licenses/MIT)

## Authors
* [Äas](https://github.com/a-kest)
* [Maurice](https://github.com/mauricesvp)
* [Roman](https://github.com/tuananhroman)
