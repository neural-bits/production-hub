import time
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from typing import Tuple
import argparse


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TrtModel:

    def __init__(self, engine_path, max_batch_size=1, dtype=np.float32):

        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine

    def allocate_buffers(self):

        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = (
                trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
            )
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def __call__(self, x: np.ndarray, batch_size=2):

        x = x.astype(self.dtype)

        np.copyto(self.inputs[0].host, x.ravel())

        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)

        self.context.execute_async(
            batch_size=batch_size,
            bindings=self.bindings,
            stream_handle=self.stream.handle,
        )
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)

        self.stream.synchronize()
        return [out.host.reshape(batch_size, -1) for out in self.outputs]


def main(engine_path: str = "mobilenetv2.engine", num_runs: int = 100, inp_shape: Tuple[int, int, int, int]=(1, 3, 224, 224)):
    trt_model = TrtModel(engine_path)

    # We're doing a first wram-up feed-forward, to:
    # - ensure CUDA context initialization
    # - skip lazy loading of some components (e.g cuDNN )
    warmp_up_data = np.random.rand(*inp_shape).astype(np.float32)
    trt_model(warmp_up_data)

    trt_timings = []
    for i in range(num_runs):
        dummy_data = np.random.rand(*inp_shape).astype(np.float32)
        st = time.time_ns()
        _ = trt_model(dummy_data)
        et = time.time_ns()
        trt_timings.append(et - st)

    with open("trt_benchmark.txt", "w") as f:
        f.write("\n".join([str(t) for t in trt_timings]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine_path", type=str, default="mobilenetv2.engine")
    parser.add_argument("--num_runs", type=int, default=100)
    parser.add_argument("--inp_shape", type=Tuple[int, int, int, int], default=(1, 3, 224, 224))
    args = parser.parse_args()
    main(args.engine_path, args.num_runs, args.inp_shape)