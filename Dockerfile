# syntax=docker/dockerfile:1
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04
WORKDIR /code
COPY ./requirements.txt .
ENV CUDA_VISIBLE_DEVICES=0
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    && \
    apt-get clean 

RUN pip install -r ./requirements.txt
