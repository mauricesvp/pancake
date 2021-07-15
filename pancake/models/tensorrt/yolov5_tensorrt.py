# import numpy as np
# import os
# import pkg_resources

# import pycuda.driver as cuda
# import pycuda.autoinit
# import torch
# import torch.nn as nn
# from typing import Type


# from pancake.logger import setup_logger
# from pancake.models.base_class import BaseModel
# from pancake.utils.general import export_onnx, check_requirements

# l = setup_logger(__name__)

# for package in ["tensorrt"]:
#     try:
#         dist = pkg_resources.get_distribution(package)
#         l.info("\u2713 " + "{} ({}) is installed".format(dist.key, dist.version))
#         import tensorrt as trt

#         trt_installed = True

#         TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
#         EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
#     except pkg_resources.DistributionNotFound:
#         l.info("\u2620 " + "{} is NOT installed".format(package))
#         trt_installed = False


# class Yolov5TRT(BaseModel):
#     def __init__(self, yolov5, weights_path):
#         # if trt not available return standard yolov5 model
#         if not trt_installed:
#             l.info("TensorRT not installed, using standard Yolov5..")
#             return yolov5

#         self.yolov5 = yolov5
#         self._required_img_size = self.yolov5._required_img_size
#         self._stride = self.yolov5._stride

#         # TRT currently only supports non-batch inference
#         self._batch_size = 1
#         x = self._required_img_size

#         input_tensor = (
#             torch.zeros(self._batch_size, 3, x, x).float().to(self.yolov5._device)
#         )

#         onnx_path = (
#             weights_path.replace(".pt", ".onnx")
#             if type(weights_path) is str
#             else weights_path[0].replace(".pt", ".onnx")
#         )

#         weights_name = onnx_path.split("/")[-1].split(".")[0]

#         l.info(f"Converting PyTorch model from weights {weights_name} to ONNX")

#         if not export_onnx(
#             self._init_export(), onnx_path, input_tensor, dynamic_axes=False
#         ):
#             l.info("Couldn't convert to ONNX, using standard Yolov5..")
#             return self.yolov5

#         # TRT init

#         # NMS on GPU
#         # max supported is 4096, if conf_thres is low, such as 0.001, use larger number.
#         self._topK = 512
#         self._keepTopK = 300  # max detections on nms

#         check_requirements(["pycuda"])

#         self.build_engine(onnx_path)
#         self.allocate_buffers()

#         # warm up
#         img = Yolov5TRT.prep_image_infer(input_tensor)
#         import time

#         for _ in range(10):
#             t1 = time.time()
#             self.infer(img)
#             print(f"Inf time: {time.time()-t1:.2f}")

#     def _init_export(self):
#         from pancake.utils.activations import Hardswish, SiLU
#         from pancake.models.yolo import Detect
#         from pancake.models.common import Conv

#         assert self.yolov5
#         model = self.yolov5.model.float()

#         for k, m in model.named_modules():
#             m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
#             if isinstance(m, Conv):  # assign export-friendly activations
#                 if isinstance(m.act, nn.Hardswish):
#                     m.act = Hardswish()
#                 elif isinstance(m.act, nn.SiLU):
#                     m.act = SiLU()
#             elif isinstance(m, Detect):
#                 m.inplace = opt.inplace
#                 m.onnx_dynamic = opt.dynamic

#         return model

#     def _init_infer(self, img_size):
#         pass

#     @staticmethod
#     def prep_image_infer(img: Type[np.array]) -> Type[np.array]:
#         """
#         Preprocesses images for inference (expanded dim (,4), half precision (fp16), normalized)

#         :param img: padded and resized image
#         :return prep_img: preprocessed image
#         """
#         if type(img) is torch.Tensor:
#             img = img.cpu().numpy()
#         img = img.astype(np.float32)
#         img /= 255.0
#         if len(img.shape) < 4:
#             img = np.expand_dims(img, axis=0)
#         img = np.ascontiguousarray(img)
#         return img

#     def infer(self, img: Type[np.array]) -> Type[np.array]:
#         """
#         :param img (np.array): resized and padded image [bs, 3, width, height]

#         :return pred (tensor): list of detections, on (,6) tensor [xyxy, conf, cls]
#                 img (tensor): preprocessed image 4d tensor [, R, G, B] (on device,
#                               expanded dim (,4), half precision (fp16))
#         """
#         # Prepare img for inference
#         img = Yolov5TRT.prep_image_infer(img)
#         # img = torch.from_numpy(img).float().numpy()

#         assert self.engine, "TRT engine hasn't been initialized yet!"

#         stream = cuda.Stream()
#         with self.engine.create_execution_context() as context:
#             self.inputs[0].host = img

#             [  # put data on gpu
#                 cuda.memcpy_htod_async(inp.device, inp.host, stream)
#                 for inp in self.inputs
#             ]
#             stream.synchronize()

#             # actual inference
#             context.execute_async_v2(
#                 bindings=self.bindings, stream_handle=stream.handle
#             )
#             stream.synchronize()

#             [  # retrieve results
#                 cuda.memcpy_dtoh_async(out.host, out.device, stream)
#                 for out in self.outputs
#             ]
#             stream.synchronize()

#             num_det = int(self.outputs[0].host[0, ...])
#             boxes = np.array(self.outputs[1].host).reshape(self._batch_size, -1, 4)[
#                 0, 0:num_det, 0:4
#             ]
#             scores = np.array(self.outputs[2].host).reshape(self._batch_size, -1, 1)[
#                 0, 0:num_det, 0:1
#             ]
#             classes = np.array(self.outputs[3].host).reshape(self._batch_size, -1, 1)[
#                 0, 0:num_det, 0:1
#             ]

#             return [np.concatenate([boxes, scores, classes], -1)], img

#     def build_engine(self, onnx_path, using_half: bool = True):
#         trt.init_libnvinfer_plugins(None, "")
#         engine_file = onnx_path.replace(".onnx", ".trt")
#         num_classes = len(self.yolov5.names)

#         # read existing trt engine
#         if os.path.exists(engine_file):
#             engine_exists = True
#             l.info(f"Found a corresponding TRT engine at {engine_file}")
#             with open(engine_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
#                 self.engine = runtime.deserialize_cuda_engine(f.read())
#                 return

#         with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
#             EXPLICIT_BATCH
#         ) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
#             builder.max_batch_size = 1  # always 1 for explicit batch
#             config = builder.create_builder_config()
#             config.max_workspace_size = 1 * 1 << 30
#             if using_half:
#                 config.set_flag(trt.BuilderFlag.FP16)

#             # Load the Onnx model and parse it in order to populate the TensorRT network.
#             with open(onnx_path, "rb") as model:
#                 if not parser.parse(model.read()):
#                     l.info(
#                         "ERROR: Failed to parse the ONNX file, using standard Yolov5.."
#                     )
#                     for error in range(parser.num_errors):
#                         l.info(parser.get_error(error))
#                     return self.yolov5

#             previous_output = network.get_output(0)
#             network.unmark_output(previous_output)

#             # slice boxes, obj_score, class_scores
#             strides = trt.Dims([1, 1, 1])
#             starts = trt.Dims([0, 0, 0])
#             bs, num_boxes, _ = previous_output.shape
#             shapes = trt.Dims([bs, num_boxes, 4])
#             boxes = network.add_slice(previous_output, starts, shapes, strides)
#             starts[2] = 4
#             shapes[2] = 1
#             obj_score = network.add_slice(previous_output, starts, shapes, strides)
#             starts[2] = 5
#             shapes[2] = num_classes
#             scores = network.add_slice(previous_output, starts, shapes, strides)

#             indices = network.add_constant(
#                 trt.Dims([num_classes]), trt.Weights(np.zeros(num_classes, np.int32))
#             )
#             gather_layer = network.add_gather(
#                 obj_score.get_output(0), indices.get_output(0), 2
#             )

#             # scores = obj_score * class_scores => [bs, num_boxes, nc]
#             updated_scores = network.add_elementwise(
#                 gather_layer.get_output(0),
#                 scores.get_output(0),
#                 trt.ElementWiseOperation.PROD,
#             )

#             # reshape box to [bs, num_boxes, 1, 4]
#             reshaped_boxes = network.add_shuffle(boxes.get_output(0))
#             reshaped_boxes.reshape_dims = trt.Dims([0, 0, 1, 4])

#             # add batchedNMSPlugin, inputs:[boxes:(bs, num, 1, 4), scores:(bs, num, 1)]
#             trt.init_libnvinfer_plugins(TRT_LOGGER, "")
#             registry = trt.get_plugin_registry()
#             assert registry
#             creator = registry.get_plugin_creator("BatchedNMS_TRT", "1")
#             assert creator
#             fc = []
#             fc.append(
#                 trt.PluginField(
#                     "shareLocation",
#                     np.array([1], dtype=np.int),
#                     trt.PluginFieldType.INT32,
#                 )
#             )
#             fc.append(
#                 trt.PluginField(
#                     "backgroundLabelId",
#                     np.array([-1], dtype=np.int),
#                     trt.PluginFieldType.INT32,
#                 )
#             )
#             fc.append(
#                 trt.PluginField(
#                     "numClasses",
#                     np.array([num_classes], dtype=np.int),
#                     trt.PluginFieldType.INT32,
#                 )
#             )
#             fc.append(
#                 trt.PluginField(
#                     "topK",
#                     np.array([self._topK], dtype=np.int),
#                     trt.PluginFieldType.INT32,
#                 )
#             )
#             fc.append(
#                 trt.PluginField(
#                     "keepTopK",
#                     np.array([self._keepTopK], dtype=np.int),
#                     trt.PluginFieldType.INT32,
#                 )
#             )
#             fc.append(
#                 trt.PluginField(
#                     "scoreThreshold",
#                     np.array([self.yolov5._conf_thres], dtype=np.float32),
#                     trt.PluginFieldType.FLOAT32,
#                 )
#             )
#             fc.append(
#                 trt.PluginField(
#                     "iouThreshold",
#                     np.array([self.yolov5._iou_thres], dtype=np.float32),
#                     trt.PluginFieldType.FLOAT32,
#                 )
#             )
#             fc.append(
#                 trt.PluginField(
#                     "isNormalized",
#                     np.array([0], dtype=np.int),
#                     trt.PluginFieldType.INT32,
#                 )
#             )
#             fc.append(
#                 trt.PluginField(
#                     "clipBoxes", np.array([0], dtype=np.int), trt.PluginFieldType.INT32
#                 )
#             )

#             fc = trt.PluginFieldCollection(fc)
#             nms_layer = creator.create_plugin("nms_layer", fc)

#             layer = network.add_plugin_v2(
#                 [reshaped_boxes.get_output(0), updated_scores.get_output(0)], nms_layer
#             )
#             layer.get_output(0).name = "num_detections"
#             layer.get_output(1).name = "nmsed_boxes"
#             layer.get_output(2).name = "nmsed_scores"
#             layer.get_output(3).name = "nmsed_classes"
#             for i in range(4):
#                 network.mark_output(layer.get_output(i))

#             # build new trt engine
#             self.engine = builder.build_engine(network, config)

#             # serialize and store the engine
#             with open(engine_file, "wb") as f:
#                 f.write(self.engine.serialize())

#                 from pancake.utils.general import file_size

#                 l.info(
#                     f"TRT engine export success, saved as {engine_file} ({file_size(engine_file):.1f} MB)"
#                 )

#     def allocate_buffers(self, is_explicit_batch=True, dynamic_shapes=[]):
#         self.inputs = []
#         self.outputs = []
#         self.bindings = []

#         class HostDeviceMem(object):
#             def __init__(self, host_mem, device_mem):
#                 self.host = host_mem
#                 self.device = device_mem

#             def __str__(self):
#                 return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

#             def __repr__(self):
#                 return self.__str__()

#         for binding in self.engine:
#             dims = self.engine.get_binding_shape(binding)
#             l.debug(f"Layer '{binding}' dim: {dims}")
#             if dims[0] == -1:
#                 assert len(dynamic_shapes) > 0
#                 dims[0] = dynamic_shapes[0]
#             size = trt.volume(dims) * self.engine.max_batch_size
#             dtype = trt.nptype(self.engine.get_binding_dtype(binding))
#             # Allocate host and device buffers
#             host_mem = cuda.pagelocked_empty(size, dtype)
#             device_mem = cuda.mem_alloc(host_mem.nbytes)
#             # Append the device buffer to device bindings.
#             self.bindings.append(int(device_mem))
#             # Append to the appropriate list.
#             if self.engine.binding_is_input(binding):
#                 self.inputs.append(HostDeviceMem(host_mem, device_mem))
#             else:
#                 self.outputs.append(HostDeviceMem(host_mem, device_mem))
