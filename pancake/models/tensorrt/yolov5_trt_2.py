import numpy as np
import itertools
import os
import pkg_resources
import time
from typing import Type

import pycuda.driver as cuda
import pycuda.autoinit
import torch
import torch.nn as nn
import torchvision

from pancake.logger import setup_logger
from pancake.models.base_class import BaseModel
from pancake.utils.general import export_onnx, check_requirements, non_max_suppression

l = setup_logger(__name__)

for package in ["tensorrt"]:
    try:
        dist = pkg_resources.get_distribution(package)
        l.info("\u2713 " + "{} ({}) is installed".format(dist.key, dist.version))
        import tensorrt as trt

        trt_installed = True
    except pkg_resources.DistributionNotFound:
        l.info("\u2620 " + "{} is NOT installed".format(package))
        trt_installed = False


class Yolov5TRT(BaseModel):
    def __init__(
        self, yolov5, engine_path: str, plugin_path: str, nms_gpu: bool, *args, **kwargs
    ):
        # if trt not available return standard yolov5 model
        check_requirements(["pycuda", "torchvision"])
        if not trt_installed:
            l.info("TensorRT not installed, using standard Yolov5..")
            return yolov5

        # store standard model
        self._yolov5 = yolov5
        self.names = self._yolov5.names
        self._required_img_size = self._yolov5._required_img_size
        self._stride = None

        # get trt engine- and plugin path
        self._engine_path = engine_path
        self._plugin_path = plugin_path
        self._nms_gpu = nms_gpu

        # --- TRT init ---
        # create a context on this device
        self.ctx = cuda.Device(0).make_context()
        self.stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        self.runtime = trt.Runtime(TRT_LOGGER)

        # loard TRT engine
        if not self.load_engine():
            return self._yolov5
        
        # allocate buffers and warm up context
        self.allocate_buffers()
        self._init_infer([self.batch_size, 3, self.input_h, self.input_w])

    def load_engine(self):
        l.info(f"Loading TRT engine from {self._engine_path}..")
        try:
            # Load trt plugin lib
            import ctypes

            ctypes.CDLL(self._plugin_path)

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
        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []

        for binding in self.engine:
            l.debug(f"Binding: {binding}, {self.engine.get_binding_shape(binding)}")
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
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)

    def _init_infer(self, img_size: None):
        # Warm up
        iterations = 20
        sum_time = 0.0

        for _ in range(iterations):
            t1 = time.time()
            self.infer(np.zeros(img_size, dtype=np.uint8))
            sum_time += time.time() - t1

        l.debug(
            f"(WARM UP) avg. inference time on {img_size}: {sum_time/iterations:.5f}"
        )

    @staticmethod
    def prep_image_infer(img: Type[np.array]) -> Type[np.array]:
        """
        Preprocesses images for inference (expanded dim (,4), half precision (fp16), normalized)

        :param img: padded and resized image
        :return prep_img: preprocessed image
        """
        if type(img) is torch.Tensor:
            img = img.cpu().numpy()
        img = img.astype(np.float32)
        img /= 255.0
        if len(img.shape) < 4:
            img = np.expand_dims(img, axis=0)
        img = np.ascontiguousarray(img)
        return img

    def prep_batches(self, imgs: Type[np.array]) -> Type[list]:
        """
        Divide imgs ndarray into batches whose sizes are compatible with the
        TRT engine input layer.

        :param imgs:
        """
        modulo = imgs.shape[0] % self.batch_size

        # fill imgs array in order to be divisible by engine batch size
        if modulo != 0:
            img_size = imgs.shape[2:]
            fills = np.zeros((modulo, 3, img_size), dtype=np.uint8)

        return np.vsplit(imgs, imgs.shape[0] / self.batch_size), modulo

    def infer(self, imgs: Type[np.array]) -> Type[np.array]:
        """
        :param img (np.array): resized and padded image [num img, 3, width, height]

        :return pred (tensor): list of detections, on (,6) tensor [xyxy, conf, cls]
                img (tensor): preprocessed image 4d tensor [, R, G, B] (on device,
                              expanded dim (,4), half precision (fp16))
        """
        # prepare imgs for inference
        imgs = Yolov5TRT.prep_image_infer(imgs)

        # prepare batches according to engine batch size
        # e.g. input [7, 3, 640, 640], engine bs = 4
        # prep_batches(input) -> [[4, 3, 640, 640], [4, 3, 640, 640]]
        # (filled with one zero entry of [1, 3, 640, 640])
        img_batches, fills = self.prep_batches(imgs)

        # infer on batches
        batched_pred = [self.infer_on_batch(batch) for batch in img_batches]

        # from [num batches, batch size, 6] to [num batches x batch size, 6]
        # (prediction per image)
        pred = list(itertools.chain.from_iterable(batched_pred))
        pred = pred[:-fills] if fills > 0 else pred  # only return non-fills
        return pred, None

    def infer_on_batch(self, img: Type[np.array]) -> Type[list]:
        """
        :param img (np.array): resized and padded image [bs, 3, width, height]

        :return pred (tensor): (one batch) list of detections, on (bs,6) tensor [xyxy, conf, cls]
                img (tensor): preprocessed image 4d tensor [, R, G, B] (on device,
                              expanded dim (,4), half precision (fp16))
        """
        assert img.shape[0] == self.batch_size, (
            f"Provided batch size ({img.shape[0]}) doesn't allign with "
            f"the engines batch size ({self.batch_size})"
        )

        # get img shapes
        img_sizes = img.shape[2:]

        # make self the active context, pushing it on top of the context stack.
        self.ctx.push()

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
        np.copyto(host_inputs[0], img.ravel())

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

        # remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()

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

    def postp_image_infer(self, output, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A tensor like [num_boxes,cx,cy,w,h,conf,cls_id]
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes tensor, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a tensor, each element is the score correspoing to box
            result_classid: finally classid, a tensor, each element is the classid correspoing to box
        """
        # get the num of boxes detected
        num = int(output[0])

        # reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, 6))[:num, :]

        # to a torch Tensor (GPU)
        pred = torch.Tensor(pred).cuda() if self._nms_gpu else torch.Tensor(pred)

        # get the boxes, scores, classid
        boxes, scores, classid = pred[:, :4], pred[:, 4], pred[:, 5]

        if self._nms_gpu:
            # choose those boxes that score > CONF_THRESH
            si = scores > self._yolov5._conf_thres
            boxes = boxes[si, :]
            scores = scores[si]
            classid = classid[si]

        # no detections found, return empty tensor
        if num == 0:
            return torch.zeros((0, 6))

        # transform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes = self.xywh2xyxy(origin_h, origin_w, boxes)

        if self._nms_gpu:
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
        else:
            scores = torch.unsqueeze(scores, 1)
            classid = torch.unsqueeze(classid, 1)

            pred = torch.cat((boxes, scores, classid), 1)
            pred = torch.unsqueeze(pred, 0)

            # do nms (CPU)
            result = non_max_suppression(
                pred,
                self._yolov5._conf_thres,
                self._yolov5._iou_thres,
                self._yolov5._classes,
                self._yolov5._agnostic_nms,
            )[0]

        return result

    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes tensor, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes tensor, each row is a box [x1, y1, x2, y2]
        """
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y
