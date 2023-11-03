#Build the Docker image (assuming the Dockerfile and train_models.py are in the current directory)
docker build -t sweta .

# Run the Docker container and mount the volume to save trained models on the host
docker run -v $(pwd)/models:/app/models sweta
