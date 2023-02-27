FROM python:3.8
RUN pip install --upgrade pip
ENV PIP_ROOT_USER_ACTION=ignore
RUN pip install poetry
RUN mkdir /app
WORKDIR /app
RUN mkdir /utils

COPY poetry.lock pyproject.toml ./
RUN poetry config virtualenvs.create false \
  && poetry install --only main --no-interaction --no-ansi
  # && poetry install --no-dev --no-interaction --no-ans
# COPY batch_sads/* /app/
COPY *.py /app/
COPY *.pkl /app/
ADD ./utils/ /app/utils
ENV PYTHONPATH=$PWD:$PYTHONPATH
# RUN ls -al 
EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "--server.headless", "true", \
            "--server.port", "8501", "st_batch.py"]
