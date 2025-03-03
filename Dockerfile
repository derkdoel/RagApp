# Use Python 3.11.8 as base image
FROM python:3.11.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy poetry files
COPY pyproject.toml poetry.lock* ./

# Install poetry and configure it
RUN pip install poetry==1.7.1 \
    && poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-interaction --no-ansi --no-root

# Copy the application files
COPY src/ ./src/
COPY ./.env /app/.env

# Expose the Streamlit port
EXPOSE 8501

# Command to run the application
CMD ["poetry", "run", "streamlit", "run", "src/streamlit_app.py"]