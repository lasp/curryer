# =============================================================================
# Test environment.
#
# Example:
#   > docker build -t curryer_test . --target test
#   > docker run -a stdout -a stderr --rm \
#       -e SPACETRACK_USER=$SPACETRACK_USER -e SPACETRACK_PSWD=$SPACETRACK_PSWD \
#       curryer_test
# -----------------------------------------------------------------------------
# Python version with which to test (must be supported and available on dockerhub)
ARG BASE_IMAGE_PYTHON_VERSION
FROM python:${BASE_IMAGE_PYTHON_VERSION:-3.11}-slim AS test

USER root

ENV BASE_DIR=/app/curryer
ENV DATA_DIR=/app/curryer/data
WORKDIR $BASE_DIR

# Create virtual environment and permanently activate it for this image
# This adds not only the venv python executable but also all installed entrypoints to the PATH
# Upgrade pip to the latest version because poetry uses pip in the background to install packages
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --upgrade pip

# Update the OS tools and install dependencies.
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    g++ \
    libgeos-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev \
    libssl-dev \
    libffi-dev \
    python3-dev \
    python3-pyproj \
    && rm -rf /var/lib/apt/lists/*

# Install poetry and add to path.
RUN pip install poetry

# Copy data files.
RUN mkdir -p $DATA_DIR/generic \
    && curl https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de430.bsp \
    --output $DATA_DIR/generic/de430.bsp
COPY data $DATA_DIR

# Copy library files.
# TODO: First copy depdencies and install python before the whole lib? Makes things faster locally.
COPY bin $BASE_DIR/bin
COPY curryer $BASE_DIR/curryer
COPY tests $BASE_DIR/tests
COPY README.md $BASE_DIR
COPY pyproject.toml $BASE_DIR

# Ensure pip is upgraded
RUN pip install --upgrade pip

# Install all dependencies (including dev deps) specified in pyproject.toml
# Copy the manually exported requirements.txt file
COPY requirements.txt $BASE_DIR

# Install dependencies using pip
RUN pip install -r $BASE_DIR/requirements.txt

# Download third-party data.
ENV CURRYER_DATA_DIR=$DATA_DIR

RUN PROJ_DOWNLOAD_DIR=$(python -c "import pyproj; print(pyproj.datadir.get_user_data_dir())") \
    && mkdir -p ${PROJ_DOWNLOAD_DIR} \
    && curl https://cdn.proj.org/us_nga_egm96_15.tif --output ${PROJ_DOWNLOAD_DIR}/us_nga_egm96_15.tif

ENTRYPOINT ["pytest", "-v", "--junitxml=junit.xml", "--disable-warnings", "tests"]

# =============================================================================
# Debug environment.
#
#
# Example for apple silicone:
# docker build --platform=linux/arm64 -t curryer_debug . --target debug
#
#
#
#
# Example:
#   > docker build -t curryer_debug . --target debug
#   > docker run --security-opt seccomp=unconfined --privileged -it --rm curryer_debug
#
#   > docker run -v .:/workspace -p 8889:8889 --security-opt seccomp=unconfined --privileged -it --rm curryer_debug
#   > cd /workspace ; export JUPYTER_CONFIG_DIR=/workspace/notebooks
#   > jupyter lab . --allow-root --ip 0.0.0.0 --port 8889 --no-browser
# -----------------------------------------------------------------------------
FROM test AS debug

# Command line tools and debug dependencies.
RUN apt-get update && apt-get install -y man less vim which tree

RUN pip install matplotlib jupyterlab cartopy basemap basemap-data-hires

CMD []
ENTRYPOINT ["/bin/bash"]

# =============================================================================
# L1A Geolocation Integration Test Case
#
# Example:
#   > docker build -t curryer_clarreo_l1a . --target demo
#   > docker run -a stdout -a stderr --rm curryer_clarreo_l1a
#       or to override with a different input file:
#   > docker run -a stdout -a stderr --rm -v `pwd`:/app/curryer/tests/data/demo curryer_clarreo_l1a
# -----------------------------------------------------------------------------
FROM test AS demo

SHELL ["/usr/bin/bash"]
WORKDIR $BASE_DIR

CMD ["tests/sit4_l1a_geolocation.py", "--demo", "--output_dir", "tests/data/demo"]
ENTRYPOINT ["python3"]