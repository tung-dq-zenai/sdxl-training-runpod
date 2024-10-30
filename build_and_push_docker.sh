#!/bin/bash

# # Enable error handling
# set -e

# Set variables
image_name="tadesscrapy/flux_training"
tag="sdxl_train_1.0.0"

# Print the image name and tag for verification
echo "Building Docker image: $image_name:$tag"

# Build the Docker image for the specific platform
docker build --platform=linux/amd64 -t "$image_name:$tag" .

# Check if the build succeeded
if [ $? -eq 0 ]; then
    echo "Docker image built successfully!"
else
    echo "Docker build failed!"
    exit 1
fi

# Push the Docker image
echo "Pushing Docker image: $image_name:$tag"
docker push "$image_name:$tag"

# Check if the push succeeded
if [ $? -eq 0 ]; then
    echo "Docker image pushed successfully!"
else
    echo "Docker push failed!"
    exit 1
fi
