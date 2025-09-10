#!/bin/bash

set -e

echo "Start building Docker image..."
docker build -t imagebrush:latest -f ./docker_manager/Dockerfile.conda .
echo "Docker image build completed!"
