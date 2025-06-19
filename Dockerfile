# Use a more full-featured Python image to ensure system dependencies are present
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that might be required by torch/torchvision for image operations
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container at /app
COPY . .

# Expose the port the app runs on
EXPOSE 5001

# Define the command to run the application
# --preload is crucial: it runs our module-level code (starting the training thread) once.
# --timeout 120 gives workers more time before being recycled, which can be good for long-running tasks.
CMD ["gunicorn", "--bind", "[::]:5001", "--workers", "1", "--threads", "2", "--timeout", "120", "--preload", "train_integrated:app"] 