FROM jupyter/minimal-notebook:python-3.10.10

USER root

COPY . ${HOME}

WORKDIR ${HOME}

RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-cache --no-interaction -vv && \
    chown -R jovyan ${HOME}

USER jovyan
