FROM python:3.12-slim-bullseye

# Install build dependencies and clean up
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc g++ && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip, install wheel and poetry in one step
RUN pip install --upgrade pip poetry

# Set working directory
WORKDIR /app

# Copy project files and install dependencies
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-root


# Copy necessary application files
COPY st_multi_batch.py .
COPY sads_logging.py .
COPY utils/ ./utils/
COPY src/ ./src/

# Copy Streamlit config file
COPY .streamlit/config.toml .streamlit/config.toml

# Set environment variables
ENV PYTHONPATH=/app
EXPOSE 8501

# Set the entry point for the Docker container
ENTRYPOINT ["streamlit", "run", "--server.headless=true", "--server.port=8501", "--server.fileWatcherType=none", "--browser.gatherUsageStats=false", "src/main.py"]
