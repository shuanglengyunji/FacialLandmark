# FacialLandmark

## Pull docker image

```
# CUDA
docker pull mxnet/python:1.5.0_gpu_cu101_py3_ub18

# CPU
docker pull mxnet/python:latest_cpu_native_py3
```

## Configure docker environment

```
# Install dependence 
apt-get -y update
apt-get install ffmpeg libsm6 libxext6 git -y

# Install insightface 
git clone https://github.com/deepinsight/insightface.git
cd ./insightface/python-package/
pip install .
```

## Run docker 
```
docker run --gpus all --rm -ti -v "$PWD":/work --ipc=host han/faciallandmark:latest
```