FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git zip unzip lsof \
    && rm -rf /var/lib/apt/lists/*

# Install AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip aws

# Copy only requirements first to leverage caching
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    python -m pip install --upgrade -r requirements.txt --no-cache-dir && \
    rm requirements.txt

# Copy the rest of the files
COPY . .

CMD ["python", "-u", "handler.py"]