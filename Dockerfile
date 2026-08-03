FROM python:3.12-slim-bookworm

ARG DEIM_REVISION=09d35d53d39ee3145a1e61e3a989b28b9468d1dd
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NO_ALBUMENTATIONS_UPDATE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/recipe
RUN git clone https://github.com/Intellindust-AI-Lab/DEIM third_party/DEIM \
    && git -C third_party/DEIM checkout "${DEIM_REVISION}" \
    && rm -rf third_party/DEIM/.git

COPY . .
RUN python -m pip install --no-cache-dir \
        torch==2.10.0 torchvision==0.25.0 \
        --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install --no-cache-dir \
        -r third_party/DEIM/requirements.txt \
        hafnia==0.7.8 \
        -e .
