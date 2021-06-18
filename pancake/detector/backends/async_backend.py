import multiprocessing as mp
import numpy as np
import traceback
import torch

from pathos.helpers import mp
from typing import Type
import queue

from .backend_dei import Backend
from pancake.detector import setup_detector

# https://stackoverflow.com/questions/25324560/strange-queue-priorityqueue-behaviour-with-multiprocessing-in-python-2-7-6
# class PQManager(mp.Manager):
#     pass

# PQManager.register(
#     "PriorityQueue", queue.PriorityQueue
# )  # Register a shared PriorityQueue

# [0, 1] - 5
# 5 % 2


def check_device_availability(devices: list):
    device_count = torch.cuda.device_count()
    devices_int = [int(dev) for dev in devices]

    for device_idx in devices_int:
        assert 0 <= device_idx and device_idx < device_count, (
            "Unallowed device number entered for 'ASYNC_BACKEND.DEVICES'!"
            f"Available devices: {[i for i in range(device_count)]}"
        )


def get_device_assignment(devices: list, num_workers: int):
    """
    Uniformly assigns each worker to a device in the list.
    """
    return [devices[i % len(devices)] for i in range(num_workers)]


def set_device_cfg(config: dict, device: str):
    """
    Changes the device in the config dictionary.
    """
    config.DEVICE = device
    return config


class AsyncBackendManager:
    """Class for multiprocessed backend calculation"""

    def __init__(
        self,
        devices: list,
        backend: Type[Backend],
        num_workers: int,
        img_q_size: int,
        det_q_size: int,
        detector_cfg: dict,
    ) -> None:
        from pancake.run import setup_logger

        self.l = setup_logger(__name__)

        self._devices = devices
        self._num_workers = num_workers
        self.backend = backend

        self.device_alloc = get_device_assignment(devices, num_workers)
        readable_alloc = {i: self.device_alloc[i] for i in range(self._num_workers)}
        self.l.info(
            f"Num workers: {self._num_workers}, Devices: {self._devices}, " 
            f"Device allocation: {readable_alloc}"
        )

        self.img_queue = mp.Queue(maxsize=img_q_size)
        self.det_queue = mp.Queue(maxsize=det_q_size)

        self.detector_cfg = detector_cfg
        self.backend.detector = None

        local_cpu_count = mp.cpu_count()
        if num_workers + 3 > local_cpu_count:
            # main_process + res_processor + async_be_manager
            self.l.warn(
                f"Potential number of subprocesses ({num_workers+3}) is higher than"
                f"actual cpu count ({local_cpu_count})! Consider lowing 'NUM_WORKERS'. \n"
                "[main_process + res_processor + async_be_manager + workers]"
            )

        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        
        self.workers = [
            mp.Process(target=self.run, args=([i])) for i in range(self._num_workers)
        ]

        for worker in self.workers:
            worker.start()

    def run(self, id):
        try:
            self.l.debug(f"Subprocess {id}")
            self._id = id
            self.backend.detector = setup_detector(
                set_device_cfg(self.detector_cfg, self.device_alloc[self._id])
            )
            while 1:
                self.l.debug("While loop")
                #try:
                imgs = self.img_queue.get(block=True, timeout=10)
                self.l.debug("Get queue")
                # except Exception:
                #     self.img_queue.close()
                #     self.det_queue.close()
                #     break
                self.det_queue.put_nowait(self.backend.detect(imgs))
        except:
            traceback.print_exc()

    def put_imgs(self, imgs: np.array) -> None:
        self.img_queue.put_nowait(imgs)

    def get_dets(self) -> (torch.Tensor, np.array):
        return self.det_queue.get_nowait()

    def res_q_empty(self) -> bool:
        return self.det_queue.empty()


def setup_async_backend(
    config: dict, backend: Type[Backend]
) -> (AsyncBackendManager, mp.Queue, mp.Queue):

    async_cfg = config.DETECTOR.ASYNC_BACKEND
    check_device_availability(async_cfg.DEVICES)

    return AsyncBackendManager(
        async_cfg.DEVICES,
        backend,
        async_cfg.NUM_WORKERS,
        async_cfg.IMG_Q_SIZE,
        async_cfg.DET_Q_SIZE,
        config,
    )
