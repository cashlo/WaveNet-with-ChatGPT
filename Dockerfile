# Use the TensorFlow GPU image as the base image
FROM tensorflow/tensorflow:latest-gpu

# Install additional libraries and dependencies
RUN apt-get update && apt-get install -y librosa-dev

# Copy the project files to the image
COPY . /app

# Set the working directory
WORKDIR /app

# Install the project dependencies
RUN pip install -r requirements.txt

# Set the default command for running the image
CMD ["python", "train.py"]