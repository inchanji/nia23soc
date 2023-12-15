FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

RUN DEBIAN_FRONTEND=noninteractive apt-get update 
RUN DEBIAN_FRONTEND=noninteractive apt-get install -yq python3-opencv git --no-install-recommends
RUN DEBIAN_FRONTEND=noninteractive apt-get install locales

COPY ./segformer-pytorch-eval /workspace/segformer-pytorch-eval
COPY ./cvt-eval /workspace/cvt-eval
COPY ./requirements.txt /workspace/requirements.txt
COPY ./dataset /workspace/dataset

COPY ./initialize.sh /workspace/initialize.sh
RUN ["chmod", "+x", "/workspace/initialize.sh"]
RUN ["sh", "/workspace/initialize.sh"]

RUN pip install -r /workspace/requirements.txt 

# WORKDIR /segformer-pytorch-eval

WORKDIR /workspace
