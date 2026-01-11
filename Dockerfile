# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/pytorch:25.09-py3

# --- OS deps ---
USER root
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    git git-lfs build-essential pkg-config python3-dev curl ca-certificates nvidia-cuda-toolkit\
 && rm -rf /var/lib/apt/lists/* \
 && git lfs install

# --- Non-root user (idempotent) ---
ARG USERNAME=appuser
ARG UID=1000
ARG GID=1000
RUN set -eux; \
    if ! getent group "${GID}" >/dev/null; then \
        groupadd -g "${GID}" "${USERNAME}"; \
    fi; \
    if ! getent passwd "${UID}" >/dev/null; then \
        useradd -m -u "${UID}" -g "${GID}" -s /bin/bash "${USERNAME}"; \
    else \
        HOME_DIR="$(getent passwd "${UID}" | cut -d: -f6)"; \
        mkdir -p "${HOME_DIR}"; chown -R "${UID}:${GID}" "${HOME_DIR}"; \
    fi

# --- Workspace, caches, env ---
WORKDIR /workspace/reasoning-from-scratch
ENV HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/workspace
RUN mkdir -p "${HF_HOME}" /workspace/reasoning-from-scratch && chown -R ${UID}:${GID} /workspace

# --- Python tools (uv only - Jupyter will be installed in venv) ---
RUN pip install --no-cache-dir uv

# --- Bring in the project (let .dockerignore keep context lean) ---
COPY --chown=${UID}:${GID} . .

# --- Install project deps if manifests exist ---
RUN set -eux; \
    if [ -f pyproject.toml ]; then \
        uv sync --frozen || uv sync; \
    elif [ -f requirements.txt ]; then \
        uv pip install --no-cache-dir -r requirements.txt; \
    else \
        echo "No dependency manifests found; skipping dependency install."; \
    fi

# --- Switch to non-root via numeric IDs (works even if name differs) ---
USER ${UID}:${GID}

# --- Jupyter Lab exposed externally via port mapping ---
EXPOSE 8888
CMD ["/workspace/reasoning-from-scratch/.venv/bin/jupyter", "lab", "--ip", "0.0.0.0", "--no-browser"]
