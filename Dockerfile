ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md NOTICE.md ./
COPY nsn/ ./nsn/
COPY neurosleepnet/ ./neurosleepnet/
COPY examples/restart_demo.py ./restart_demo.py
RUN pip install --no-cache-dir .
ENV NSN_DATA_DIR=/data NSN_NAMESPACE=docker_agent
VOLUME ["/data"]
CMD ["python", "restart_demo.py"]
