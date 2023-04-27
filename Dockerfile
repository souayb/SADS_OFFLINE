# FROM python:3.8
# RUN pip install --upgrade pip
# ENV PIP_ROOT_USER_ACTION=ignore
# RUN pip install poetry
# RUN mkdir /app
# WORKDIR /app
# RUN mkdir /utils

# COPY poetry.lock pyproject.toml ./
# RUN poetry config virtualenvs.create false \
#   && poetry install --only main --no-interaction --no-ansi

# COPY *.py /app/
# COPY *.pkl /app/
# ADD ./utils/ /app/utils
# ENV PYTHONPATH=$PWD:$PYTHONPATH
# # RUN ls -al
# EXPOSE 8501

# ENTRYPOINT ["streamlit", "run", "--server.headless", "true", \
#             "--server.port", "8501", "st_batch.py"]

# ___________________________________________________

# FROM python:3.8

# # Install Poetry
# RUN pip install --upgrade pip && \
#     pip install poetry

# # Copy project files and install dependencies
# WORKDIR /app
# COPY pyproject.toml poetry.lock ./
# RUN poetry config virtualenvs.create false \
#     && poetry install --no-interaction --no-ansi --no-root

# # Copy application files
# COPY . .

# # Set environment variables
# ENV PYTHONPATH=/app
# EXPOSE 8501

# # Set the entry point for the Docker container
# ENTRYPOINT ["streamlit", "run", "--server.headless=true", "--server.port=8501", "st_batch.py"]

# ------------------------------------------------------

# # Use an official Python runtime as a parent image
# FROM python:3.8-slim-buster

# # Set the working directory to /app
# WORKDIR /app

# # Copy only the files necessary for installation
# COPY pyproject.toml poetry.lock ./

# # Install system dependencies and Poetry
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends gcc && \
#     pip install poetry && \
#     poetry config virtualenvs.create false

# # Install application dependencies
# RUN poetry install --no-dev --no-interaction --no-ansi

# # Copy the rest of the application files
# COPY . .

# # Expose port 8501 for Streamlit
# EXPOSE 8501

# # Set the entry point for the Docker container
# ENTRYPOINT ["streamlit", "run", "--server.headless=true", "--server.port=8501", "st_batch.py"]



# FROM python:3.8-slim-buster

# # Install Poetry
# RUN pip install --upgrade pip && \
#     pip install poetry

# # Copy only the necessary files for dependency installation
# WORKDIR /app
# COPY pyproject.toml poetry.lock ./
# RUN apt-get update && \
#     apt-get install -y --no-install-recommends gcc && \
#     pip install poetry && \
#     poetry config virtualenvs.create false

# # Copy only the necessary files for application execution
# COPY st_batch.py .
# COPY utils/ ./utils/

# # Set environment variables
# ENV PYTHONPATH=/app
# EXPOSE 8501

# # Set the entry point for the Docker container
# ENTRYPOINT ["streamlit", "run", "--server.headless=true", "--server.port=8501", "st_batch.py"]


# FROM python:3.8-slim-buster

# # Install build-essential and Poetry
# RUN apt-get update && \
#     apt-get install -y build-essential && \
#     pip install --upgrade pip && \
#     pip install poetry

# # Copy only the necessary files for dependency installation
# WORKDIR /app
# COPY pyproject.toml poetry.lock ./

# # Install application dependencies
# RUN poetry config virtualenvs.create false && \
#     poetry install --no-dev --no-interaction --no-ansi

# # Copy only the necessary files for application execution
# COPY st_batch.py .
# COPY model.pkl .
# COPY sads_logging.py .
# COPY utils/ ./utils/

# # Set environment variables
# ENV PYTHONPATH=/app
# EXPOSE 8501

# # Set the entry point for the Docker container
# ENTRYPOINT ["streamlit", "run", "--server.headless=true", "--server.port=8501", "st_batch.py"]



FROM python:3.8

# Install Poetry
RUN pip install --upgrade pip && \
    pip install poetry

# Copy project files and install dependencies
WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-root

# Copy only the necessary files for application execution
COPY st_multi_batch.py .
COPY model.pkl .
COPY sads_logging.py .
COPY utils/ ./utils/
# Copy Streamlit config file
COPY .streamlit/config.toml .streamlit/config.toml


# Set environment variables
ENV PYTHONPATH=/app
EXPOSE 8501

# Set the entry point for the Docker container
ENTRYPOINT ["streamlit", "run", "--server.headless=true", "--server.port=8501", "st_multi_batch.py"]
