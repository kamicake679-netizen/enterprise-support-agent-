# ==============================================================================
# 星云智联（NebulaLink）企业级智能客服与知识库 Agent - Docker 镜像构建文件
# 基础镜像：官方 python:3.11-slim（海外/全球网络通用，无国内源硬编码）
# ==============================================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

# 先安装 Python 依赖以最大化复用构建缓存
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 复制应用代码、前端页面、知识库源文档与启动脚本
COPY app ./app
COPY web ./web
COPY scripts ./scripts
COPY data/docs ./data/docs
COPY .env.example ./.env.example

# 兼容 Windows 换行符并赋予容器启动脚本可执行权限
RUN sed -i 's/\r$//' /app/scripts/docker_entrypoint.sh && \
    chmod +x /app/scripts/docker_entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; port = os.environ.get('PORT', '8000'); urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=5)" || exit 1

ENTRYPOINT ["/app/scripts/docker_entrypoint.sh"]
