# Detector

## YOLOv5 Simple
  A very simple detector using a pretrained model provided by YOLOv5.
  
  **Configuration options:** (under <code>YOLO_SIMPLE:</code>)
  
  | Parameter               | Example Values   | Description         |
  | ---------------------   | ----------------- | ------------------- |
  | <code>size</code> | "s", "m", "l" or "x"         | Yolo model size

## YOLOv5 Custom
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

### YOLOv5 TensorRT
  >NVIDIA® TensorRT™ is an SDK for high-performance deep learning inference. It includes a deep learning inference optimizer and runtime that delivers low latency and high throughput for deep learning inference applications.

  TensorRT bear the potential of significantly speeding up the inference. For that purpose, we investigated the usage of a TensorRT engine for predict the bounding boxes instead of the standard model. 

  In order to be able to enable TensorRT inference with YOLOv5, you have to generate a specific engine with respective library with [this repository](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5). A detailed How-To is given in the repo's description. 

  After you have successfully transformed the original model, you need to specify the engine's and plugin library's path and set ```trt``` to *True*.