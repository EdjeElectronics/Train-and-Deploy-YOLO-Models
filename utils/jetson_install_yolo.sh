#!/bin/bash

# Get sudo permission
sudo -v

# Confirm user is running Jetpack 6.1 (L4T 36.4)
if [ -f /etc/nv_tegra_release ]; then
    L4T_RELEASE=$(sed -n 's/.*R\([0-9]*\) (release).*/\1/p' /etc/nv_tegra_release)
    L4T_REVISION=$(sed -n 's/.*REVISION: \([0-9.]*\).*/\1/p' /etc/nv_tegra_release)
else
    echo "nv_tegra_release file not found, so this is not a Jetson system. Exiting script."
    exit 1
fi

if [[ "$L4T_RELEASE" == "36" && "$L4T_REVISION" == "4"* ]]; then
    echo "JetPack version is correct (6.1 / L4T $L4T_RELEASE.$L4T_REVISION)"
else
    echo "Wrong version of NVIDIA JetPack is installed."
    echo "Detected L4T: R$L4T_RELEASE Revision $L4T_REVISION"
    echo "Please install JetPack 6.1 from the SD Card Image downloadable at: https://developer.nvidia.com/embedded/jetpack-sdk-61"
    echo "Exiting Ultralytics installation script due to unsupported JetPack version."
    exit 1
fi


# Install Ultralytics and other required packages
pip install ultralytics
pip install onnx onnxslim
pip install numpy==1.26.4
pip install opencv-python==4.10.0.82


# Install special versions of torch and torchvision (compatible with Jetpack 6.1)
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl


# Install cusparelt fix
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install libcusparselt0 libcusparselt-dev


# Install onnxruntime-gpu
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl

echo "Ultralytics YOLO installation successful!"
