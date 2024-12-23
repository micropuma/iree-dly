cmake -G Ninja -B ../iree-cuda/ -S . \
    -DIREE_TARGET_BACKEND_CUDA=ON \
    -DCMAKE_CXX_FLAGS="-Wno-error=dangling-pointer" \
    -DIREE_HAL_DRIVER_CUDA=ON \
    -DIREE_BUILD_PYTHON_BINDINGS=ON  \
    -DPython3_EXECUTABLE="$(which python3)" \
    -DIREE_CUDA_LIBDEVICE_PATH=/usr/local/cuda-12.1/nvvm/libdevice/libdevice.10.bc