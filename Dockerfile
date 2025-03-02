FROM python:3.10-slim

LABEL maintainer="Multi-Platform Content Generator Team"
LABEL description="Container for Multi-Platform Content Generator application"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libx11-xcb1 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xinerama0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Ensure scripts are executable
RUN chmod +x *.py *.sh *.command

# Run data setup
RUN python setup_data.py
RUN python sync_data.py

# Set environment variable
ENV PYTHONUNBUFFERED=1
ENV QT_QPA_PLATFORM=offscreen

# Command to run on container start
ENTRYPOINT ["python", "app.py"]