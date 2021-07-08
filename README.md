# pancake :pancakes: - Panorama Camera Car Tracking

_pancake_ is an application for panorama camera car tracking. It comes with a simple and modular program design facilitating easy implementation and application of different techniques regarding panorama stitching, object detection and object tracking. 

**Following features are included:**
* Straight forward implementation and application of state-of-the-art panorama stitching, object detection and object tracking technologies
* Include a discretionary number of image streams of various source types
* Several options for result visualization 
* Optional database logging of vehicle tracks with Sqlite3 
* Modular structure for extension of new functionalities and approaches

## Table of Contents
- [pancake :pancakes: - Panorama Camera Car Tracking](#pancake-pancakes---panorama-camera-car-tracking)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
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
    - [Authors](#authors)

## Installation

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
## Modules
### Backend
### Object Detectoin
### Object Tracking
[Back to ToC](#table-of-contents)
## Further Notes

### Samples
[drive](https://drive.google.com/drive/folders/1Y8FiPtxZiQrv7BrF05uWrDqHdZ8bA11h)

### Google Colab

[Google Colab for training Yolov5 models on custom data](https://colab.research.google.com/drive/1xtMJhFlp0cB9S2_irIkHAjJ9_6Tol-g9?usp=sharing)

[Google Colab for executing pancake](https://colab.research.google.com/drive/1NGkIHqXnOPeZqV1RbvcGoQ_DTjOgpGxC?usp=sharing)


### TensorRT
[External repo for generating TRT engines](https://github.com/adujardin/tensorrtx/tree/trt8_yolov5_support/yolov5)

[Back to ToC](#table-of-contents)

### Authors
* [Äas](https://github.com/a-kest)
* [Maurice](https://github.com/mauricesvp)
* [Roman](https://github.com/tuananhroman)