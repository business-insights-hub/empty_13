# AgriBot - Kənd Təsərrüfatı RAG System
# Dockerfile for containerized deployment

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-simple.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-simple.txt

# Copy application code
COPY . .

# Create directories if they don't exist
RUN mkdir -p templates static/css dataset

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
