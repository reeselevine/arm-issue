# Coherency Issue on ARM GPU

This repository includes Vulkan code that should reproduce a coherency issue related to reordering stores. The core code of the test is:

Thread 1:
```
atomic_store(x, 1, memory_order_relaxed);
atomic_store(x, 2, memory_order_relaxed);
```

Thread 2:
```
uint r0 = atomic_load(x, memory_order_relaxed);
uint r1 = atomic_load(x, memory_order_relaxed);
```

where `x` is a memory location initialized to 0.

In this test, it should be impossible for `r1` to load an earlier write than `r0`; for example, having `r0` load 2 but `r1` load 1 or 0. However, on the G78 Arm GPU running in a Pixel 6, we do observe this reordering some of the time.

## How this repository works
Running only two threads on the GPU is very unlikely to observe this issue. These test shaders therefore run many instances of the test in parallel, which is accomplished by having each thread run Thread 1's instructions on one memory location, and running Thread 2's instructions on some other memory location. Threads are paired up so that each test instance is fully executed.

The shaders were originally written in OpenCL, and then compiled to SPIR-V using [clspv](https://github.com/google/clspv). Both the original and the compiled shaders are included here, with the OpenCL code annotated to show how the test works. Additionally, this repository depends on [easyvk](https://github.com/reeselevine/easyvk) for the Vulkan setup code, as well as Vulkan being available on the system. The binary, `TestRunner`, can be built using cmake:

```
mkdir build
cd build
cmake ..
cmake --build .
```

Make sure that when cloning, the `--recurse-submodules` option is included, to pick up the easyvk submodule.
