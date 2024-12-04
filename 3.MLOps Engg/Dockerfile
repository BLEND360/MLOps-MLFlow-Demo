# Use a lightweight Python image as the base
FROM python:3.9-slim

# Set a working directory inside the container
WORKDIR /app

# Install system dependencies required for LightGBM
RUN apt-get update && apt-get install -y libgomp1

# Copy the requirements.txt file into the container
COPY ../requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the inference script and modules into the container
COPY inference.py .
COPY modules /app/modules

# Set environment variables (if any are required)
ENV MLFLOW_TRACKING_URI="http://localhost:5000/"

# Run the inference script that pulls the model from MLflow and serves it
CMD ["python", "inference.py"]
