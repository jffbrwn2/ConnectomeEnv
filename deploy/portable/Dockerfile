FROM python:3.11-slim-bookworm@sha256:f06d01d4ba3f8ad5af45a3ec87df19c1d317d8e8961edc0fabda61811af6f7cf
RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libegl-mesa0 libgl1 libgl1-mesa-dri libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/connectome
# Copy only the installable environment; never credentials, data or checkpoints.
COPY deploy/portable/requirements.lock /opt/connectome/requirements.lock
RUN pip install --no-cache-dir -r requirements.lock
COPY pyproject.toml README.md LICENSE ./
COPY src/proofreading ./src/proofreading
RUN pip install --no-cache-dir --no-deps .
ARG SOURCE_REVISION=unavailable
ENV CONNECTOME_SOURCE_REVISION=$SOURCE_REVISION \
    CONNECTOME_RENDERER=software \
    CONNECTOME_MESH_CACHE=/cache/meshes \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1
RUN mkdir -p /cache && chmod 1777 /cache
WORKDIR /work
ENTRYPOINT ["python", "-m", "proofreading.experiment"]
