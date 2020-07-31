FROM buildpack-deps:focal-scm@sha256:41ebc088feff8aaac8f19f64bda1ad4f8313dbe102da2b6ca504ae774207e19f AS base
LABEL maintainer "https://github.com/weiji14"
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Initiate docker container with user 'jovyan'
ARG NB_USER=jovyan
ARG NB_UID=1000
ENV NB_USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

# Setup conda
ENV CONDA_DIR ${HOME}/.conda
ENV NB_PYTHON_PREFIX ${CONDA_DIR}
ENV MINICONDA_VERSION 4.8.3

RUN cd /tmp && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-py38_${MINICONDA_VERSION}-Linux-x86_64.sh && \
    echo "d63adf39f2c220950a063e0529d4ff74 *Miniconda3-py38_${MINICONDA_VERSION}-Linux-x86_64.sh" | md5sum -c - && \
    /bin/bash Miniconda3-py38_${MINICONDA_VERSION}-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-py38_${MINICONDA_VERSION}-Linux-x86_64.sh && \
    $CONDA_DIR/bin/conda config --system --prepend channels conda-forge && \
    $CONDA_DIR/bin/conda config --system --set auto_update_conda false && \
    $CONDA_DIR/bin/conda config --system --set show_channel_urls true && \
    $CONDA_DIR/bin/conda config --system --set pip_interop_enabled true && \
    $CONDA_DIR/bin/conda clean --all --quiet --yes && \
    $CONDA_DIR/bin/conda init --verbose

# Setup $HOME directory with correct permissions
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}
WORKDIR ${HOME}

# Change to interactive bash shell, so that `conda activate` works
SHELL ["/bin/bash", "-ic"]

# Install dependencies in environment.yml file using conda
COPY environment.yml ${HOME}
RUN conda env create -n deepicedrain -f environment.yml && \
    conda clean --all --yes && \
    conda info && \
    conda list -n deepicedrain

# Install dependencies in poetry.lock using poetry
COPY pyproject.toml ${HOME}/
COPY poetry.lock ${HOME}/
RUN conda activate deepicedrain && \
    poetry install && \
    rm --recursive ${HOME}/.cache/pip && \
    poetry env info && \
    poetry show

# Install jupyterlab extensions
RUN conda activate deepicedrain && \
    jupyter labextension install @pyviz/jupyterlab_pyviz && \
    jupyter labextension install dask-labextension && \
    jupyter labextension list

# Setup DeepBedMap virtual environment properly
RUN conda activate deepicedrain && \
    python -m ipykernel install --user --name deepicedrain && \
    jupyter kernelspec list --json

# Copy remaining files to $HOME
COPY --chown=1000:1000 . ${HOME}


FROM base AS app

# Run Jupyter Lab via poetry in conda environment
EXPOSE 8888
RUN echo -e '#!/bin/bash -i\nset -e\nconda activate deepicedrain\npoetry run "$@"' > .entrypoint.sh && \
    chmod +x .entrypoint.sh
ENTRYPOINT ["./.entrypoint.sh"]
CMD ["jupyter", "lab", "--ip", "0.0.0.0"]
