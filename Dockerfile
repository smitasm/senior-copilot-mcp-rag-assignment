# One image, three services. alarm-api, ticketing-api and gui (which also
# hosts the orchestrator and the three MCP servers in-process - see
# orchestrator/mcp_client.py for why) all run from this same image; which
# one a given container becomes is decided entirely by the `command:` in
# docker-compose.yml, not by anything baked in here.
#
# Pinned to python:3.13-slim specifically because 3.13 is the exact version
# this project was developed and tested against (3.12 was unavailable on the
# development machine; 3.14 was ruled out early for third-party wheel
# availability) - matching it here avoids any "works on my machine, not in
# Docker" surprise from a Python version mismatch.
FROM python:3.13-slim

WORKDIR /app

# Install dependencies in their own layer, before copying the rest of the
# source, so an ordinary code change doesn't invalidate pip's cache and
# force a full reinstall on every rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Runs as a non-root user - a small, low-effort hardening step with no
# functional downside here (nothing in this project needs root).
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

# No CMD/ENTRYPOINT: docker-compose.yml sets the right command per service.
