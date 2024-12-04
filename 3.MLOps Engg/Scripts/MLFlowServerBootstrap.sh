#!/bin/bash
# Define variables
imageName="mlflow-ubuntu"

# Container names, ports, and host port mappings
declare -A containers
containers=(
    ["dev"]=5000
    ["test"]=5001
    ["prod"]=5002
)

# Function to check if Docker is installed
check_docker_installation() {
    if command -v docker &> /dev/null; then
        echo "Docker is installed."
        return 0
    else
        echo "Docker is not installed."
        return 1
    fi
}

# Function to install Docker on Ubuntu (assuming the instance is Ubuntu)
install_docker() {
    echo "Installing Docker..."
    
    if [ -n "$sudoPassword" ]; then
        echo "$sudoPassword" | sudo -S apt-get update
        echo "$sudoPassword" | sudo -S apt-get install -y apt-transport-https ca-certificates curl software-properties-common
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo -S apt-key add -
        echo "$sudoPassword" | sudo -S add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
        echo "$sudoPassword" | sudo -S apt-get update
        echo "$sudoPassword" | sudo -S apt-get install -y docker-ce
    else
        sudo apt-get update
        sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
        sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
        sudo apt-get update
        sudo apt-get install -y docker-ce
    fi

    echo "Docker installed successfully."
}

# Function to check if Docker daemon is running
check_docker_daemon() {
    if systemctl is-active --quiet docker; then
        echo "Docker daemon is running."
    else
        echo "Docker daemon is not running. Starting Docker..."
        if [ -n "$sudoPassword" ]; then
            echo "$sudoPassword" | sudo -S systemctl start docker
        else
            sudo systemctl start docker
        fi
        echo "Docker daemon started."
    fi
}

# Function to open ports in the firewall (if applicable)
open_ports() {
    for port in "${containers[@]}"; do
        if [ -n "$sudoPassword" ]; then
            echo "$sudoPassword" | sudo -S ufw allow $port/tcp
        else
            sudo ufw allow $port/tcp
        fi
        echo "Port $port opened in the firewall."
    done
}

# Check if Docker is installed and running
if ! check_docker_installation; then
    install_docker
fi

check_docker_daemon

# Open required ports in the firewall
open_ports

# Create the Dockerfile content
dockerfileContent="# Use a slimmer base image
FROM python:3.8-slim

# Set environment variables to avoid user prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install necessary packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install MLflow
RUN pip install --no-cache-dir --upgrade pip mlflow

# Set the working directory
WORKDIR /app

# Expose the default MLflow port
EXPOSE 5000

# Default command to run MLflow server
CMD [\"mlflow\", \"server\", \"--host\", \"0.0.0.0\"]"

# Create the Dockerfile in the current directory
echo "$dockerfileContent" > Dockerfile
echo "Dockerfile created in the current directory."

# Check if the Docker image already exists
imageExists=$(docker images -q $imageName)

if [ -z "$imageExists" ]; then
    echo "Image $imageName not found. Building the Docker image..."
    
    # Build the Docker image
    if [ -n "$sudoPassword" ]; then
        echo "$sudoPassword" | sudo -S docker build -t $imageName .
    else
        sudo docker build -t $imageName .
    fi

    echo "Docker image $imageName has been built successfully."
else
    echo "Docker image $imageName already exists. Skipping build."
fi

# Start MLflow servers for dev, test, and prod environments
for env in "${!containers[@]}"; do
    hostPort=${containers[$env]}
    containerName="mlflow-container-$env"

    # Check if the Docker container already exists
    containerExists=$(sudo docker ps -a -q -f "name=$containerName")

    if [ -z "$containerExists" ]; then
        echo "Container $containerName not found. Running the container on port $hostPort..."
        
        # Run the Docker container
        if [ -n "$sudoPassword" ]; then
            echo "$sudoPassword" | sudo -S docker run -d -p $hostPort:5000 --name $containerName $imageName
        else
            sudo docker run -d -p $hostPort:5000 --name $containerName $imageName
        fi

        echo "Container $containerName is running."
        echo "MLflow UI for $env should be accessible at $remoteIP:$hostPort"
    else
        echo "Container $containerName already exists. Skipping run."
    fi
done

# Remove the Dockerfile after everything is done
rm Dockerfile
echo "Dockerfile has been removed."
