# Build the Docker image
docker build -t wanderly-email-app .

# Run the container
docker run -p 8501:8501 wanderly-email-app

# Alternatively, with docker-compose
docker-compose up