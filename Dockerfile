FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git zip unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage caching
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install --upgrade -r requirements.txt --no-cache-dir && \
    rm requirements.txt

# Copy the rest of the files
COPY . .

ENV NUM_TOKENS=2
ENV DISABLE_AUTO_CAPTIONS=True

CMD ["python", "-u", "handler.py"]