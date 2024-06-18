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

ENV BASE_DIR /app/curryer
ENV DATA_DIR $BASE_DIR/data
WORKDIR $BASE_DIR

# Create virtual environment and permanently activate it for this image
# This adds not only the venv python executable but also all installed entrypoints to the PATH
# Upgrade pip to the latest version because poetry uses pip in the background to install packages
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --upgrade pip

# Update the OS tools and install dependencies.
RUN apt-get update
RUN apt-get install -y curl

# Install poetry and add to path.
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH="$PATH:/root/.local/bin"

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
RUN poetry install

# Download third-party data.
ENV CURRYER_DATA_DIR=$DATA_DIR

RUN PROJ_DOWNLOAD_DIR=$(python -c "import pyproj; print(pyproj.datadir.get_user_data_dir())") \
    && mkdir -p ${PROJ_DOWNLOAD_DIR} \
    && curl https://cdn.proj.org/us_nga_egm96_15.tif --output ${PROJ_DOWNLOAD_DIR}/us_nga_egm96_15.tif

ENTRYPOINT pytest \
    -v \
    --junitxml=junit.xml \
    --disable-warnings \
    tests

# =============================================================================
# Debug environment.
#
# Example:
#   > docker build -t curryer_debug . --target debug
#   > docker run --security-opt seccomp=unconfined --privileged -it --rm curryer_debug
# -----------------------------------------------------------------------------
FROM test AS debug

# Command line tools and debug dependencies.
RUN apt-get install -y man less vim which tree

CMD []
ENTRYPOINT ["/bin/bash"]

# =============================================================================
# L1A Geolocation Integration Test Case
#
# Example:
#   > docker build -t curryer_clarreo_l1a. --target clarreo_l1a
#   > docker run -a stdout -a stderr --rm curryer_clarreo_l1a
#       or to override with a different input file:
#   > docker run -a stdout -a stderr --rm -v `pwd`:/app/curryer/tests/data/demo curryer_clarreo_l1a
# -----------------------------------------------------------------------------
FROM test AS demo

SHELL ["/usr/bin/bash"]
WORKDIR $BASE_DIR

CMD ["tests/sit4_l1a_geolocation.py", "--demo", "--output_dir", "tests/data/demo"]
ENTRYPOINT ["python3"]