FROM buildpack-deps:jammy-scm@sha256:69a05e44c60e1a1002f2d9699f0418c640a3eadc89dcdfc9dbe87db7e7d87887 AS base
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

# Setup mamba
ENV MAMBA_DIR ${HOME}/.mamba
ENV NB_PYTHON_PREFIX ${MAMBA_DIR}
ENV MAMBAFORGE_VERSION 4.14.0-0

RUN cd /tmp && \
    wget --quiet https://github.com/conda-forge/miniforge/releases/download/${MAMBAFORGE_VERSION}/Mambaforge-${MAMBAFORGE_VERSION}-Linux-x86_64.sh && \
    echo "d47b78b593e3cf5513bafbfa6a51eafcd9f0e164c41c79c790061bb583c82859 *Mambaforge-${MAMBAFORGE_VERSION}-Linux-x86_64.sh" | sha256sum -c - && \
    /bin/bash Mambaforge-${MAMBAFORGE_VERSION}-Linux-x86_64.sh -f -b -p $MAMBA_DIR && \
    rm Mambaforge-${MAMBAFORGE_VERSION}-Linux-x86_64.sh && \
    $MAMBA_DIR/bin/mamba clean --all --quiet --yes && \
    $MAMBA_DIR/bin/mamba init --verbose

# Setup $HOME directory with correct permissions
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}
WORKDIR ${HOME}

# Change to interactive bash shell, so that `mamba activate` works
SHELL ["/bin/bash", "-ic"]

# Install dependencies in environment-linux-64.lock file using mamba
COPY environment.yml ${HOME}
COPY environment-linux-64.lock ${HOME}
RUN mamba create --name deepicedrain --file environment-linux-64.lock && \
    mamba clean --all --yes && \
    mamba info && \
    mamba list -n deepicedrain

# Install dependencies in poetry.lock using poetry
COPY pyproject.toml ${HOME}/
COPY poetry.lock ${HOME}/
RUN mamba activate deepicedrain && \
    poetry install && \
    rm --recursive ${HOME}/.cache/pip && \
    poetry env info && \
    poetry show

# Setup DeepBedMap virtual environment properly
RUN mamba activate deepicedrain && \
    python -m ipykernel install --user --name deepicedrain && \
    jupyter kernelspec list --json

# Copy remaining files to $HOME
COPY --chown=1000:1000 . ${HOME}


FROM base AS app

# Run Jupyter Lab via poetry in conda environment
EXPOSE 8888
RUN echo -e '#!/bin/bash -i\nset -e\nmamba activate deepicedrain\npoetry run "$@"' > .entrypoint.sh && \
    chmod +x .entrypoint.sh
ENTRYPOINT ["./.entrypoint.sh"]
CMD ["jupyter", "lab", "--ip", "0.0.0.0"]
