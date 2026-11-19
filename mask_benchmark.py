"""RGB difference, threshold and 4x4 mask reduction: CPU, CUDA, OpenCL."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from numba import njit

SIZES = [(64, 48), (640, 480), (1920, 1080), (3840, 2160), (7680, 4320)]
FIXED_THRESHOLD = 20
DOWNSAMPLE = 4
INPUT_TYPES = ('rgb',)


# One output pixel per thread; each block reports any detected movement.
GPU_BODY = """
    int i = OUTPUT_INDEX;
    int out_height = (height + 3) / 4;
    if (i >= out_width * out_height) return;
    int oy = i / out_width, ox = i % out_width;
    unsigned char detected = 0;
    for (int dy = 0; dy < 4; ++dy) {
        int y = oy * 4 + dy;
        if (y >= height) continue;
        for (int dx = 0; dx < 4; ++dx) {
            int x = ox * 4 + dx;
            if (x >= width) continue;
            int pixel = (y * width + x) * 3;
            unsigned char av = ((unsigned int)a[pixel] + a[pixel+1] + a[pixel+2]) / 3;
            unsigned char bv = ((unsigned int)b[pixel] + b[pixel+1] + b[pixel+2]) / 3;
            unsigned char d = av >= bv ? av - bv : bv - av;
            detected |= d > 20;
        }
    }
    mask[i] = detected;
"""


@njit
def cpu_mask(a, b):
    height, width = a.shape[:2]
    mask = np.empty(((height + 3)//4, (width + 3)//4), dtype=np.uint8)
    for oy in range(mask.shape[0]):
        for ox in range(mask.shape[1]):
            detected = False
            for y in range(oy*4, min(oy*4+4, height)):
                for x in range(ox*4, min(ox*4+4, width)):
                    av = (np.uint16(a[y, x, 0]) + a[y, x, 1] + a[y, x, 2]) // 3
                    bv = (np.uint16(b[y, x, 0]) + b[y, x, 1] + b[y, x, 2]) // 3
                    difference = av - bv if av >= bv else bv - av
                    detected |= difference > FIXED_THRESHOLD
            mask[oy, ox] = detected
    return mask


class CPU:
    name = 'cpu'

    def upload(self, a):
        return a

    def pull(self, a):
        return np.asarray(a)

    def sync(self):
        pass

    def mask(self, a, b):
        return cpu_mask(a, b)

    def info(self):
        return {'library': 'Numba compiled single-threaded loop', 'dtype': 'uint8'}

class CUDA:
    name = 'cuda'

    def __init__(self):
        import cupy as cp
        self.cp = cp
        if cp.cuda.runtime.getDeviceCount() == 0:
            raise RuntimeError('No CUDA device')
        self.kernel = cp.RawKernel(
                'extern "C" __global__ void movement_mask(const unsigned char *a, '
                'const unsigned char *b, unsigned char *mask, int width, int height, int out_width) {'
                + GPU_BODY.replace('OUTPUT_INDEX', 'blockIdx.x * blockDim.x + threadIdx.x') + '}',
                'movement_mask')

    def upload(self, a):
        return self.cp.asarray(a)

    def pull(self, a):
        return self.cp.asnumpy(a)

    def sync(self):
        self.cp.cuda.get_current_stream().synchronize()

    def mask(self, a, b):
        height, width = a.shape[:2]
        shape = ((height + 3)//4, (width + 3)//4)
        mask = self.cp.empty(shape, dtype=self.cp.uint8)
        self.kernel(((mask.size + 255)//256,), (256,),
                    (a, b, mask, np.int32(width), np.int32(height), np.int32(shape[1])))
        return mask

    def info(self):
        cp = self.cp
        p = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        name = p['name']
        if isinstance(name, bytes):
            name = name.decode()
        return {'device': name, 'library': 'CuPy fused RGB RawKernel', 'runtime': cp.cuda.runtime.runtimeGetVersion(), 'driver': cp.cuda.runtime.driverGetVersion(), 'dtype': 'uint8'}

class OpenCL:
    name = 'opencl'

    def __init__(self, device_name=None):
        import pyopencl as cl
        import pyopencl.array as cl_array
        devices = [d for p in cl.get_platforms() for d in p.get_devices() if d.type & cl.device_type.GPU]
        if device_name:
            devices = [d for d in devices if device_name.lower() in d.name.lower()]
        if len(devices) != 1:
            raise RuntimeError(f'Select exactly one OpenCL GPU with --opencl-device. Matching devices: {[d.name for d in devices]}')
        self.device = devices[0]
        self.cl_array = cl_array
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)
        program = cl.Program(self.context,
                '__kernel void movement_mask(__global const unsigned char *a, '
                '__global const unsigned char *b, __global unsigned char *mask, '
                'int width, int height, int out_width) {'
                + GPU_BODY.replace('OUTPUT_INDEX', 'get_global_id(0)') + '}').build()
        self.kernel = program.movement_mask

    def upload(self, a):
        return self.cl_array.to_device(self.queue, a)

    def pull(self, a):
        return a.get()

    def sync(self):
        self.queue.finish()

    def mask(self, a, b):
        height, width = a.shape[:2]
        shape = ((height + 3)//4, (width + 3)//4)
        mask = self.cl_array.empty(self.queue, shape, np.uint8)
        self.kernel(self.queue, (mask.size,), None, a.data, b.data, mask.data,
                    np.int32(width), np.int32(height), np.int32(shape[1]))
        return mask

    def info(self):
        d = self.device
        return {'device': d.name, 'vendor': d.vendor, 'platform': d.platform.name, 'driver': d.driver_version, 'opencl_version': d.version, 'library': 'PyOpenCL fused kernel', 'dtype': 'uint8'}

def timed(backend, function, repeats, warmup):
    for _ in range(warmup):
        result = function()
        backend.sync()
        del result
    samples = []
    for _ in range(repeats):
        backend.sync()
        start = perf_counter()
        result = function()
        backend.sync()
        samples.append((perf_counter() - start) * 1000)
        del result
    return samples


def host_pipeline(backend, a, b):
    mask = backend.mask(backend.upload(a), backend.upload(b))
    return backend.pull(mask)


def make_frames(width, height):
    rng = np.random.default_rng(1701 + width + height)
    a = rng.integers(20, 61, (height, width, 3), dtype=np.uint8)
    b = (a.astype(np.int16) + rng.integers(-3, 4, a.shape, dtype=np.int16)).astype(np.uint8)
    h, w = max(4, height//6), max(4, width//8)
    y, x = height//3, width//3
    dy, dx = max(1, h//4), max(1, w//4)
    a[y:y+h, x:x+w] = 220
    b[y+dy:y+dy+h, x+dx:x+dx+w] = 220
    return a, b


def verify(backend, a, b):
    # Independent vectorized reference; widen RGB sums to prevent overflow.
    gray_a = (a.sum(axis=2, dtype=np.uint16) // a.shape[2]).astype(np.int16)
    gray_b = (b.sum(axis=2, dtype=np.uint16) // b.shape[2]).astype(np.int16)
    full_mask = np.abs(gray_a - gray_b) > FIXED_THRESHOLD
    height, width = full_mask.shape
    padded = np.pad(full_mask, ((0, (-height) % 4), (0, (-width) % 4)))
    expected = padded.reshape((height+3)//4, 4, (width+3)//4, 4).any(axis=(1, 3)).astype(np.uint8)
    actual = host_pipeline(backend, a, b)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.uint8
    return dict(threshold=FIXED_THRESHOLD, output_width=expected.shape[1],
                output_height=expected.shape[0], foreground_pixels=int(np.count_nonzero(expected)))


def checks(backend):
    # Each possible pair of RGB-derived intensities gets its own output block, so errors cannot hide in a reduction.
    av, bv = np.meshgrid(np.arange(256, dtype=np.uint8), np.arange(256, dtype=np.uint8))
    a = np.repeat(np.repeat(av, 4, axis=0), 4, axis=1)
    b = np.repeat(np.repeat(bv, 4, axis=0), 4, axis=1)
    verify(backend, np.repeat(a[:, :, None], 3, axis=2), np.repeat(b[:, :, None], 3, axis=2))
    rng = np.random.default_rng(42)
    for shape in ((1, 1, 3), (17, 19, 3), (8, 12, 3)):
        verify(backend, rng.integers(0, 256, shape, dtype=np.uint8),
               rng.integers(0, 256, shape, dtype=np.uint8))
    # A detection at every location must survive, including partial edge blocks.
    for y in range(5):
        for x in range(7):
            a = np.zeros((5, 7, 3), np.uint8)
            b = a.copy()
            b[y, x] = (63, 0, 0)  # Integer mean 21: just above threshold.
            verify(backend, a, b)
            b[y, x] = (62, 0, 0)  # Integer mean 20: at threshold.
            verify(backend, a, b)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backends', nargs='+', choices=['cpu', 'cuda', 'opencl'], default=['cpu', 'cuda', 'opencl'])
    p.add_argument('--opencl-device')
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--output', type=Path, default=Path('mask_benchmark'))
    args = p.parse_args()
    if args.repeats < 1 or args.warmup < 1:
        p.error('repeats and warmup must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    backends, unavailable = [CPU()], {}
    for name in dict.fromkeys(args.backends):
        if name == 'cpu':
            continue
        try:
            backends.append(CUDA() if name == 'cuda' else OpenCL(args.opencl_device))
        except Exception as error:
            unavailable[name] = f'{type(error).__name__}: {error}'
            print(f'{name}: UNAVAILABLE: {unavailable[name]}', flush=True)
    # Missing runtimes are reported; computation/correctness failures stop the run.
    for backend in backends:
        checks(backend)
    rows, samples, verification = [], [], []
    stages = ['pipeline_resident', 'pipeline_host_to_host']
    for input_type in INPUT_TYPES:
        for width, height in SIZES:
            a, b = make_frames(width, height)
            for backend in backends:
                ga, gb = backend.upload(a), backend.upload(b)
                check = verify(backend, a, b)
                verification.append(dict(width=width, height=height, backend=backend.name, input_type=input_type,
                                         input_sha256=hashlib.sha256(a.tobytes()+b.tobytes()).hexdigest(), **check))
                functions = [lambda: backend.mask(ga, gb),
                             lambda: host_pipeline(backend, a, b)]
                for stage, fn in zip(stages, functions):
                    values = timed(backend, fn, args.repeats, args.warmup)
                    key = dict(width=width, height=height, backend=backend.name, input_type=input_type, stage=stage)
                    rows.append(dict(**key, median_ms=float(np.median(values)),
                                     p10_ms=float(np.percentile(values, 10)), p90_ms=float(np.percentile(values, 90))))
                    samples.extend(dict(**key, repeat=i, ms=v) for i, v in enumerate(values))
                    print(f'{input_type:4} {width}x{height} {backend.name:6} {stage:23} {rows[-1]["median_ms"]:.4f} ms', flush=True)
    versions = {}
    for name in ('numpy', 'numba', 'cupy-cuda12x', 'cupy-cuda13x', 'pyopencl'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    environment = dict(measured_at_utc=datetime.now(timezone.utc).isoformat(),
        platform=platform.platform(), cpu=platform.processor(), python=sys.version, versions=versions,
        backends={b.name: b.info() for b in backends}, unavailable=unavailable,
        repeats=args.repeats, warmup=args.warmup, cpu_threads=1,
        contract='uint8 RGB; intensity = (R+G+B)//3; abs difference > 20; any in 4x4 blocks; uint8 mask 0/1',
        input_types=INPUT_TYPES,
        fixed_threshold=FIXED_THRESHOLD, downsample=DOWNSAMPLE,
        timing='synchronized API wall latency; warm repeated inputs; allocations included; no file I/O',
        host_output='full uint8 binary mask in CPU RAM',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for filename, data in [('results.csv', rows), ('samples.csv', samples)]:
        with (args.output/filename).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    for filename, data in [('environment.json', environment), ('correctness.json', verification)]:
        (args.output/filename).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    report = ['# Rozdíl a prahování naměřené výsledky', '',
              'Mediány latence v ms.', '',
              '| Vstup → maska | Backend | Celek rezidentní | Celek CPU→CPU | Zrychlení CPU→CPU |',
              '|---|---|---:|---:|---:|']
    for input_type in INPUT_TYPES:
        for width, height in SIZES:
            cpu_ms = next(r['median_ms'] for r in rows if
                          (r['width'], r['height'], r['backend'], r['input_type'], r['stage']) ==
                          (width, height, 'cpu', input_type, 'pipeline_host_to_host'))
            for backend in backends:
                values = {r['stage']: r['median_ms'] for r in rows if
                          (r['width'], r['height'], r['backend'], r['input_type']) == (width, height, backend.name, input_type)}
                numbers = ' | '.join(f'{values[s]:.4f}' for s in stages)
                report.append(f'| {input_type.upper()} {width} × {height} → {(width+3)//4} × {(height+3)//4} | {backend.name} | {numbers} | '
                              f'{cpu_ms/values["pipeline_host_to_host"]:.2f}× |')
    (args.output/'vysledky.md').write_text('\n'.join(report), encoding='utf-8')


if __name__ == '__main__':
    main()
