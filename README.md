# pancake :pancakes: - Panorama Camera Car Tracking

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
      - [Troubleshoot](#troubleshoot)
    - [Pipenv, Virtualenv](#pipenv-virtualenv)
      - [Troubleshoot](#troubleshoot-1)
    - [(Optional) Additional Software](#optional-additional-software)
      - [NVIDIA Driver, CUDA Toolkit](#nvidia-driver-cuda-toolkit)
      - [OpenCV GPU](#opencv-gpu)
  - [Usage](#usage)
    - [Quickstart](#quickstart)
    - [Configurations](#configurations)
        - [SOURCE](#source)
        - [ROI](#roi)
  - [Modules](#modules)
    - [Backend](#backend)
    - [Object Detectoin](#object-detectoin)
    - [Object Tracking](#object-tracking)
  - [Further Notes](#further-notes)
    - [Samples](#samples)
    - [Google Colab](#google-colab)
    - [TensorRT](#tensorrt)
    - [Used Third Party Repos](#used-third-party-repos)
    - [Authors](#authors)

<!------------------------- Installation --------------------------->
## Installation

### Poetry

>Poetry is arguably Python's most sophisticated dependency management option available today. Poetry goes far beyond dependencies, with features like generating .lock files, generating project scaffolding, and a ton of configuration options, all of which are handled via a simple CLI. If you're unsure how to cleanly and effectively structure and manage your Python projects, do yourself a favor and use Poetry. [Source](https://hackersandslackers.com/python-poetry-package-manager/)

1. Make sure ```Poetry``` is installed:

  ```bash 
    poetry --version
  ```
 How to install ```Poetry```: https://python-poetry.org/docs/

2. Check if ```Python3.8``` is installed:

  ```bash
    which python3.8
  ```
  How to install ```Python3.8```: https://linuxize.com/post/how-to-install-python-3-8-on-ubuntu-18-04/

3. Clone our repo into a desired location: (we will refer to the target location with ```$TARGET_DIR$```)

  ```bash
    cd ~/$TARGET_DIR$
    
    // either via HTTPS
    git clone https://github.com/mauricesvp/pancake.git

    // or via SSH
    git clone git@github.com:mauricesvp/pancake.git
  ```

4. Afterwards, navigate to the pancake location and install the dependencies:
   
  ```bash
    cd ~/$TARGET_DIR$/pancake
    poetry install
  ``` 

5. Finally, activate the virtual environment and run the main script:

  ```bash
    poetry shell
    python pancake/run.py
  ```

**For more information on basic Poetry usage refer to: https://python-poetry.org/docs/basic-usage/**

#### Troubleshoot

* When trying to install the dependencies:
>The current project's Python requirement (X.X.XX) is not compatible with some of the required packages Python requirement:
  
1. Navigate to pancake directory and delete the ```poetry.lock```:
  ```bash
    cd ~/$TARGET_DIR$/pancake
    sudo rm poetry.lock
  ```

2. Then, let poetry know we want to use ```Python3.8```: (find out the location via ```which python3.8```)
  ```bash
    poetry env use /$PATH_TO_PY38$
  ```

3. Now, try to install the dependencies again:
  ```bash
    poetry install
  ```

<br>

### Pipenv, Virtualenv

We definitely recommend to use _Poetry_ as python package manager. Still, in case you want to use _Virtualenv_ or _Pipenv_, we provide a ```requirements.txt``` and  ```dev-requirements.txt```.

1. Create a Pipenv or Virtualenv with ```Python3.8```
2. Now, activate your python environment and install the dependencies:
  ```bash 
    source $PATH_TO_ENV$/bin/activate     # Pipenv
    # or
    workon $VENV_NAME$                    # Virtualenv

    pip install -r requirements.txt       # Base packages
    pip install -r dev-requirements.txt   # Development packages
  ```
3. Have fun cooking up some pancakes:
  ```
    cd ~/$TARGET_DIR$/pancake/pancake
    python run.py
  ```

#### Troubleshoot

<br>

### (Optional) Additional Software
(_for proper speed_)
#### NVIDIA Driver, CUDA Toolkit
(_for GPU model inference_)
#### OpenCV GPU
(_for GPU image manipulation_)

<!------------------------- Usage --------------------------->
## Usage

### Quickstart
After you have followed the steps from the [installation](#installation), simply start the main script with:
```bash
python ~/pancake/pancake/run.py
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


</details>

<!-- Data -->
<details>
  <summary><b>Data</b></summary>
  <br>
  Specify a data source/s to retrieve images from as well as a region of interest to be applied on the latter. 

  ##### SOURCE
  There are several types of sources one can whip into the pancake dough. Essentially, the quantity of provided sources determine the number of frames to be assembled into a panorama.

  ##### ROI
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
### Backend
### Object Detectoin
### Object Tracking
[Back to ToC](#table-of-contents)

<!------------------------- Further Notes --------------------------->
## Further Notes

### Samples
[drive](https://drive.google.com/drive/folders/1Y8FiPtxZiQrv7BrF05uWrDqHdZ8bA11h)

### Google Colab

[Google Colab for training Yolov5 models on custom data](https://colab.research.google.com/drive/1xtMJhFlp0cB9S2_irIkHAjJ9_6Tol-g9?usp=sharing)

[Google Colab for executing pancake](https://colab.research.google.com/drive/1NGkIHqXnOPeZqV1RbvcGoQ_DTjOgpGxC?usp=sharing)


### TensorRT
[External repo for generating TRT engines](https://github.com/adujardin/tensorrtx/tree/trt8_yolov5_support/yolov5)

[Back to ToC](#table-of-contents)

### Used Third Party Repos
* **Yolov5**, Ultralytics: https://github.com/ultralytics/yolov5
* **DeepSORT**: https://github.com/nwojke/deep_sort
* **Centroid Tracker**: https://gist.github.com/adioshun/779738c3e28151ffbb9dc7d2b13c2c0a
### Authors
* [Äas](https://github.com/a-kest)
* [Maurice](https://github.com/mauricesvp)
* [Roman](https://github.com/tuananhroman)