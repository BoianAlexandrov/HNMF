FROM python:3.10-slim
WORKDIR /home/python

# install lib
COPY pyproject.toml .
COPY src/ ./src/
RUN pip install -U pip
RUN pip install .

RUN rm -r build/ src/ pyproject.toml

COPY dls_model.py dls_model.py
COPY green_model.py green_model.py

# RUN rm -r build/ tmd_rag.egg-info tmd_rag pyproject.toml
