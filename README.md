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
      - [Sources](#sources)
      - [Result Processing](#result-processing)
      - [Database](#database)
  - [Modules](#modules)
    - [Panorama Stitching](#panorama-stitching)
    - [Object Detectoin](#object-detectoin)
    - [Object Tracking](#object-tracking)
  - [Further Notes](#further-notes)
    - [Authors](#authors)
    - [Samples](#samples)
    - [Google Colab](#google-colab)
    - [TensorRT](#tensorrt)

## Installation

## Usage

### Quickstart
After you have followed the steps [above](#installation), simply start the script with:
```bash
python ~/pancake/pancake/run.py
```


<details>
  <summary>How do I dropdown?</summary>
  <br>
  This is how you dropdown.
</details>

### Configurations
All of the pancake ingredients can simply be specified in the designated ```pancake.yaml```. What can be
#### Sources

#### Result Processing

#### Database
[Back to ToC](#table-of-contents)
## Modules
### Panorama Stitching
### Object Detectoin
### Object Tracking
[Back to ToC](#table-of-contents)
## Further Notes
### Authors
* [Äas](https://github.com/a-kest)
* [Maurice](https://github.com/mauricesvp)
* [Roman](https://github.com/tuananhroman)

### Samples
[drive](https://drive.google.com/drive/folders/1Y8FiPtxZiQrv7BrF05uWrDqHdZ8bA11h)

### Google Colab

[Google Colab for training Yolov5 models on custom data](https://colab.research.google.com/drive/1xtMJhFlp0cB9S2_irIkHAjJ9_6Tol-g9?usp=sharing)

[Google Colab for executing pancake](https://colab.research.google.com/drive/1NGkIHqXnOPeZqV1RbvcGoQ_DTjOgpGxC?usp=sharing)


### TensorRT
[External repo for generating TRT engines](https://github.com/adujardin/tensorrtx/tree/trt8_yolov5_support/yolov5)

[Back to ToC](#table-of-contents)