FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Avoiding user interaction with tzdata
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y --no-install-recommends pkg-config libpng-dev libfreetype6-dev unzip ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt /workspace/
ARG IMAGE_TYPE=full


RUN pip install -r requirements.txt
RUN pip install runpod


RUN pip freeze | grep numpy && pip uninstall -y numpy || echo "numpy not found"
RUN pip install "numpy<1.24"

COPY . /workspace

CMD ["python", "motion_inbetween_handler.py"]