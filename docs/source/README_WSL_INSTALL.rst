# Install on WSL (Training & Command-Line Inference Only)

## Installing WSL

Ensure you install WSL based on Ubuntu 24.04

## Setting up WSL
```
sudo apt install wslu
```

# Install on Windows (Command-Line Inference & MATLAB Inference)

CUDA Version (Python): 12.5
cuDNN Version (Python): 9.3.0
TensorRT Version: 10.11.0

Installing CUDA and cuDNN: Through Python / pip
## Installing TensorRT 10.11.0
1. Dowload ZIP file: [Download Link](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.11.0/zip/TensorRT-10.11.0.33.Windows.win10.cuda-12.9.zip)
2. Unpack ZIP
3. Copy `TensorRT-10.11.0.33` to `C:\Program Files`
4. Add `TensorRT-10.11.0.33\lib` to system PATH
4. Add `TensorRT-10.11.0.33\bin` to system PATH