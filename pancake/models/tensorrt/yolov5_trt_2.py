""" YOLOv5 TensorRT Class """
from typing import Type, List, Union

import numpy as np
import itertools
import pkg_resources
import time

import torch
import torchvision

from pancake.logger import setup_logger
from pancake.models.base_class import BaseModel
from pancake.utils.general import (
    check_requirements,
    xywh2xyxy,
)
from pancake.utils.function_profiler import profile

l = setup_logger(__name__)

for package in ["tensorrt"]:
    try:
        dist = pkg_resources.get_distribution(package)
        l.info("\u2713 " + "{} ({}) is installed".format(dist.key, dist.version))
        import tensorrt as trt

        trt_installed = True

        check_requirements(["pycuda", "torchvision"])
        import pycuda.driver as cuda
        import pycuda.autoinit
    except:
        l.info("\u2620 " + "{} is NOT installed".format(package))
        trt_installed = False


class Yolov5TRT(BaseModel):
    def __init__(
        self, yolov5, engine_path: str, plugin_path: str, device: str, *args, **kwargs
    ):
        """ YOLOv5 TensorRT Class

        Description:
            In order to use this class it is required to first have CUDA Toolkit, \
            CuDNN and TensorRT installed. Furthermore, you need to generate a \
            TensorRT engine and plugin library with this external repo:

            https://github.com/wang-xinyu/tensorrtx/tree/master/yolov5

        Args:
            yolov5 (YOLOCustomDetector): Instance of YOLOCustomDetector
            engine_path (str): Path of the TRT engine
            plugin_path (str): Path of the TRT plugin
            device (str): Device number

        Raises:
            ModuleNotFoundError: If TensorRT library is not installed
            ModuleNotFoundError: When invalid device index was provided
            ModuleNotFoundError: When loading of the engine and plugin failed
        """    
        # if trt not available return standard yolov5 model
        if not trt_installed:
            l.info("TensorRT not installed, using standard Yolov5..")
            raise ModuleNotFoundError

        # store standard model
        self._yolov5 = yolov5
        self.names = self._yolov5.names
        self._required_img_size = self._yolov5._required_img_size
        self._stride = None

        # get trt engine- and plugin path
        self._engine_path = engine_path
        self._plugin_path = plugin_path

        # create a context on this device
        try:
            device = int(device)
        except:
            l.info(
                f"Given device {device} was not a device index! Using standard Yolov5.."
            )
            raise ModuleNotFoundError

        self.ctx = cuda.Device(device).make_context()
        self.stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(TRT_LOGGER)

        # loard TRT engine
        if not self.load_engine():
            raise ModuleNotFoundError

        # allocate buffers and warm up context
        self.allocate_buffers()
        self._init_infer([self.batch_size, 3, self.input_h, self.input_w])

    def load_engine(self) -> bool:
        """ Loads the plugin library and TRT engine.

        Returns:
            bool: success flag
        """        
        l.info(f"Loading TRT engine from {self._engine_path}..")
        try:
            # Load trt plugin lib
            import ctypes

            ctypes.CDLL(self._plugin_path)
        except Exception as e:
            l.info(
                f"Error occured while loading TRT plugin from {self._plugin_path}: \n"
                f"{e} \n"
                f"Using standard Yolov5 model.."
            )
            return False

        try:
            # Deserialize the engine from file
            with open(self._engine_path, "rb") as f:
                self.engine = self.runtime.deserialize_cuda_engine(f.read())

            # get execution context and fixed batch size
            self.context = self.engine.create_execution_context()
            self.batch_size = self.engine.max_batch_size
            return True
        except Exception as e:
            l.info(
                f"Error occured while loading engine from {self._engine_path}: \n"
                f"{e} \n"
                f"Using standard Yolov5 model.."
            )
            return False

    def allocate_buffers(self, is_explicit_batch=True, dynamic_shapes=[]):
        """ Allocates memory on the GPU according to the provided engine.

        Args:
            is_explicit_batch (bool, optional): Explicit batch flag. Defaults to True.
            dynamic_shapes (list, optional): Dynamic input shapes. Defaults to [].
        """        
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []

        for binding in self.engine:
            l.debug(
                f"Binding name: {binding}, shape: {self.engine.get_binding_shape(binding)}"
            )
            size = (
                trt.volume(self.engine.get_binding_shape(binding))
                * self.engine.max_batch_size
            )
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append the device buffer to device bindings.
            self.bindings.append(int(cuda_mem))

            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                self.input_w = self.engine.get_binding_shape(binding)[-1]
                self.input_h = self.engine.get_binding_shape(binding)[-2]

                assert (
                    self.input_w == self._required_img_size
                    and self.input_h == self._required_img_size
                ), (
                    "Provided img_size in config "
                    f"({self._required_img_size}x{self._required_img_size}) "
                    f"doesn't match with the TRT engines input size "
                    f"({self.input_h}x{self.input_w})!"
                )

                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)

    def _init_infer(self, img_size):
        """ Warms up the GPU.

        Args:
            img_size (Tuple): Image shape
        """        
        # Warm up
        iterations = 20
        sum_time = 0.0

        for _ in range(iterations):
            t1 = time.time()
            self.infer(np.zeros(img_size, dtype=np.float32))
            sum_time += time.time() - t1

        l.debug(
            f"(WARM UP) avg. inference time on {img_size}: {sum_time/iterations:.5f}"
        )

    @staticmethod
    def prep_image_infer(img: Union[torch.Tensor, np.array]) -> np.array:
        """ Preprocesses images for inference \
            (expanded dim (,4), half precision (fp16), normalized)

        Args:
            img (Union[torch.Tensor, np.array]): Resized and padded image [c, w, h] or [bs, c, w, h]

        Returns:
            np.array: Normalized image
        """        
        if type(img) is torch.Tensor:
            img = img.cpu().numpy(np.float32)
        img = img.astype(np.float32)
        img /= 255.0
        if len(img.shape) < 4:
            img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img)
        return img

    def prep_batches(self, imgs: np.ndarray) -> List[np.ndarray]:
        """ Divides a contiguous array of images in [bs, c, w, h] into a List \
            of arrays [bs, c, w, h] with batch sizes compatible with the \
            TRT engine input layer.

        Args:
            imgs (np.array): Images in a single batch

        Returns:
            List[np.ndarray]: List of image batches with processable batch size
        """        
        modulo = imgs.shape[0] % self.batch_size

        # fill imgs array in order to be divisible by engine batch size
        if modulo != 0:
            fills = np.zeros(
                (self.batch_size - modulo, 3, imgs.shape[2], imgs.shape[3]),
                dtype=np.float32,
            )
            imgs = np.concatenate((imgs, fills))

        batched_imgs = np.vsplit(imgs, imgs.shape[0] / self.batch_size)
        return batched_imgs, modulo

    def infer(self, imgs: np.array) -> List[torch.Tensor]:
        """ Inference wrapper

        Args:
            imgs (np.array): Resized and padded image [c, w, h] or [bs, c, w, h]

        Returns:
            List[torch.Tensor]: List of detections, on (,6) tensor [xyxy, conf, cls]
        """             
        # prepare imgs for inference
        # t1 = time.time()
        imgs = Yolov5TRT.prep_image_infer(imgs)
        # l.debug(f"prep_image_infer: {time.time()-t1:.5f}")

        # prepare batches according to engine batch size
        # e.g. input [7, 3, 640, 640], engine bs = 4
        # prep_batches(input) -> [[4, 3, 640, 640], [4, 3, 640, 640]]
        # (filled with one zero entry of [1, 3, 640, 640])
        img_batches, fills = self.prep_batches(imgs)

        # infer on batches
        # t1 = time.time()
        batched_pred = [self.infer_on_batch(batch) for batch in img_batches]
        # l.debug(f"infer_on_batch: {time.time()-t1:.5f}")

        # from [num batches, batch size, 6] to [num batches x batch size, 6]
        # (prediction per image)
        pred = list(itertools.chain.from_iterable(batched_pred))
        pred = pred[:-fills] if fills > 0 else pred  # only return non-fills
        return pred, None

    def infer_on_batch(self, img: np.array) -> List[torch.Tensor]:
        """ Batched inference

        Args:
            img (np.array): Resized and padded image batch [1, c, w, h]

        Returns:
            list: List of tensors with the prediction results [xyxy, conf, cls]
        """        
        assert img.shape[0] == self.batch_size, (
            f"Provided batch size ({img.shape[0]}) doesn't allign with "
            f"the engines batch size ({self.batch_size})"
        )

        # get img shapes
        img_sizes = img.shape[2:]

        # restore essential components
        stream, context, engine, bindings = (
            self.stream,
            self.context,
            self.engine,
            self.bindings,
        )
        host_inputs, cuda_inputs, host_outputs, cuda_outputs = (
            self.host_inputs,
            self.cuda_inputs,
            self.host_outputs,
            self.cuda_outputs,
        )

        # copy input image to host buffer
        # np.copyto(host_inputs[0], img.ravel())
        host_inputs[0] = img.ravel()
        # start = time.time()

        # transfer input data to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)

        # run inference.
        context.execute_async(
            batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle
        )

        # transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)

        # synchronize the stream
        stream.synchronize()

        # end = time.time()
        # l.debug(f"Inf (pure) time: {end-start:.4f}")

        # here we use the first row of output in that batch_size = 1
        output = host_outputs[0]

        # do postprocess
        pred = [
            self.postp_image_infer(
                output[i * 6001 : (i + 1) * 6001], img_sizes[0], img_sizes[1]
            )
            for i in range(self.batch_size)
        ]
        return pred

    def postp_image_infer(self, output: np.ndarray, origin_h, origin_w) -> torch.Tensor:
        """ Reshapes the model output to interpretable shape then applies NMS

        Args:
            output (np.ndarray): A tensor like [num_boxes, cx, cy, w, h, conf, cls_id]
            origin_h ([type]): height of original image
            origin_w ([type]): width of original image

        Returns:
            torch.Tensor: tensor with the prediction results [xyxy, conf, cls]
        """        
        # get the num of boxes detected
        num = int(output[0])

        # reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, 6))[:num, :]

        # to a torch Tensor (GPU)
        pred = torch.Tensor(pred).cuda()

        # get the boxes, scores, classid
        boxes, scores, classid = pred[:, :4], pred[:, 4], pred[:, 5]

        # choose those boxes that score > CONF_THRESH
        si = scores > self._yolov5._conf_thres
        boxes = boxes[si, :]
        scores = scores[si]
        classid = classid[si]

        # no detections found, return empty tensor
        if num == 0:
            return torch.zeros((0, 6))

        # transform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes = xywh2xyxy(boxes)

        # do nms (GPU)
        indices = torchvision.ops.nms(
            boxes, scores, iou_threshold=self._yolov5._iou_thres
        ).cpu()

        result_boxes = boxes[indices, :].cpu()
        result_scores = scores[indices].cpu()
        result_classid = classid[indices].cpu()

        result_scores = torch.unsqueeze(result_scores, 1)
        result_classid = torch.unsqueeze(result_classid, 1)

        result = torch.cat((result_boxes, result_scores, result_classid), 1)

        return result
