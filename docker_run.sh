IMAGE_NAME="swetaimage"
docker build -t $IMAGE_NAME .
docker run -v $(pwd)/models:/app/models $IMAGE_NAME
