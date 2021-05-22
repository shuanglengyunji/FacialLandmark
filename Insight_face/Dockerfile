FROM mxnet/python:1.5.0_gpu_cu101_py3_ub18

RUN apt-get -y update
RUN apt-get install ffmpeg libsm6 libxext6 git -y

RUN git clone https://github.com/deepinsight/insightface.git
RUN cd ./insightface/python-package/ && pip install .
RUN rm -rf insightface
