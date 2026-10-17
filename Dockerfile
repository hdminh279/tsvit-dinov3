# Hugging Face Spaces / Production Container Deployment
FROM python:3.12-slim

WORKDIR /app

# Install system libraries for GDAL / rasterio
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt .

# Install python wheels with PyTorch CPU
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy application source code
COPY . .

# Expose standard Streamlit port
EXPOSE 7860

# Configure Streamlit for cloud container
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false

CMD ["streamlit", "run", "app/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
