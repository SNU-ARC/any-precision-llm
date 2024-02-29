FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
MAINTAINER SangLyul Cho <chosanglyul@gmail.com>

RUN apt update && apt upgrade -y && apt install -y pip ninja-build vim

WORKDIR /home/any-precision-llm
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

WORKDIR any_precision/modules/kernels
COPY any_precision/modules/kernels .
# TODO support sm60 and sm61 (Pascal)
RUN TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0" python3 setup.py sdist bdist_wheel
RUN pip install dist/any_precision_ext-0.0.0-cp310-cp310-linux_x86_64.whl

WORKDIR ../../..
COPY . .
RUN python3 setup.py sdist bdist_wheel
RUN pip install dist/any_precision_llm-0.0.0-py3-none-any.whl
# RUN mv any_precision/cache/packed ../cache
# RUN mv demo.py ../demo.py

WORKDIR ..
# RUN rm -rf any-precision-llm

CMD /bin/bash
