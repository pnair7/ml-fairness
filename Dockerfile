# syntax=docker/dockerfile:1

FROM ucsdets/datahub-base-notebook
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
