FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Avoiding user interaction with tzdata
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends pkg-config libpng-dev libfreetype6-dev unzip ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt /workspace/
COPY doublefinalmotionref263.npy /workspace/
ARG IMAGE_TYPE=full

COPY . /workspace
RUN pip install openai runpod vllm
RUN pip install -r requirements.txt
RUN pip install groq


COPY . /workspace

EXPOSE 8000
COPY motion_inbetween_handler.py /workspace/



CMD ["python", "motion_inbetween_handler.py"]