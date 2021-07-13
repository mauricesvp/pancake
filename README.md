# pancake :pancakes: - Panorama Camera Car Tracking

[<img src="https://user-images.githubusercontent.com/26842284/125350856-a71c9a80-e35f-11eb-9302-b9e648d37e6e.gif" width="90%">](https://user-images.githubusercontent.com/26842284/125350856-a71c9a80-e35f-11eb-9302-b9e648d37e6e.gif)

_pancake_ is an application for panorama camera car tracking. It comes with a simple and modular program design facilitating easy implementation and application of different techniques regarding panorama stitching, object detection and object tracking. 

**Following features are included:**
* Straight forward implementation and application of state-of-the-art panorama stitching, object detection and object tracking technologies
* Include a discretionary number of image streams of various source types
* Several options for result visualization 
* Optional database logging of vehicle tracks with Sqlite3 
* Modular structure for extension of new functionalities and approaches

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
    - [Object Detection](#object-detection)
    - [Object Tracking](#object-tracking)
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
  | **Image**             | Path to single image                                     | ```"../samples/r45/1c/1621796022.9767.jpg"``` | |
  | **Video**             | Path to single video                                     | ```"../samples/output.avi"```| |
  | **Sequence of Images**| Path to directory holding several images                 | ```"../samples/r45/1c"```| |
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
</details>

<!-- Detector -->
<details>
  <summary><b>Detector</b></summary>
  <br>
</details>

<!-- Tracker -->
<details>
  <summary><b>Tracker</b></summary>
  <br>
</details>

<!-- Result Processing -->
<details>
  <summary><b>Result Processing</b></summary>
  <br>

</details>

<br>

[Back to ToC](#table-of-contents)

<!------------------------- Modules --------------------------->
## Modules

<img width="800" height="%" src="/gitimg/app_structure.png">

### Backend
### Object Detection
### Object Tracking
[Back to ToC](#table-of-contents)

<!------------------------- Further Notes --------------------------->
## Further Notes

### Google Colab

[Google Colab for training Yolov5 models on custom data](https://colab.research.google.com/drive/1xtMJhFlp0cB9S2_irIkHAjJ9_6Tol-g9?usp=sharing)

[Google Colab for executing pancake](https://colab.research.google.com/drive/1NGkIHqXnOPeZqV1RbvcGoQ_DTjOgpGxC?usp=sharing)


### TensorRT
[External repo for generating TRT engines](https://github.com/adujardin/tensorrtx/tree/trt8_yolov5_support/yolov5)


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
