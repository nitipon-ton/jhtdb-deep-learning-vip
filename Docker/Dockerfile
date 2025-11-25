#
# Public Dockerfile for jhtdb-ml
#
# This image:
#   - Uses Miniconda as a base
#   - Creates the `jhtdb-ml` conda environment from `environment.yml`
#   - Sets `/workspace` as the working directory with the repository copied in
#   - Leaves the default command as an interactive shell
#

FROM continuumio/miniconda3:latest

ENV DEBIAN_FRONTEND=noninteractive

# Where the project will live in the container
WORKDIR /workspace

# ---------------------------------------------------------------------------
# Create the conda environment from environment.yml
# ---------------------------------------------------------------------------

# Copy only the environment file first so Docker can cache this layer
COPY environment.yml /tmp/environment.yml

RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy

# Make sure the environment is on PATH
ENV CONDA_DEFAULT_ENV=jhtdb-ml
ENV PATH=/opt/conda/envs/jhtdb-ml/bin:$PATH

# Optional: install some basic OS tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Copy the repository into the image (for users who don't mount a volume)
COPY . /workspace

# Default to an interactive shell; users can override with `docker run ... <command>`
CMD ["bash"]

