@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM 切换到项目根目录
cd /d "%~dp0"

echo ============================================================
echo   星云智联（NebulaLink）企业级智能客服 Agent - Windows 一键启动
echo ============================================================
echo.

REM 1. 检查 .env 配置文件
if not exist ".env" (
    echo [错误] 未检测到 .env 配置文件！
    echo        请先执行：copy .env.example .env
    echo        并在 .env 中填入有效的 LLM_API_KEY 与 EMBEDDING_API_KEY。
    echo.
    pause
    exit /b 1
)

REM 2. 检查并激活虚拟环境 .venv
if not exist ".venv\Scripts\python.exe" (
    echo [错误] 未检测到 Python 虚拟环境 ^(.venv^)。
    echo        请先在项目根目录执行以下命令创建环境并安装依赖：
    echo          python -m venv .venv
    echo          .venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo [1/4] 正在激活虚拟环境 .venv ...
call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [错误] 激活 .venv 虚拟环境失败，请检查环境完整性。
    pause
    exit /b 1
)

REM 3. 检查模拟业务数据库是否已初始化
if not exist "data\business.db" (
    echo [2/4] 未检测到 data\business.db，正在初始化模拟订单与工单数据库...
    python app\db\mock_db.py
    if errorlevel 1 (
        echo [错误] 模拟业务数据库初始化失败！
        pause
        exit /b 1
    )
) else (
    echo [2/4] 业务数据库 data\business.db 已就绪。
)

REM 4. 检查 ChromaDB 向量库是否已初始化
if not exist "data\chroma_db\chroma.sqlite3" (
    echo [3/4] 未检测到向量知识库 data\chroma_db，正在执行文档切片与向量入库...
    python -m app.rag.ingest
    if errorlevel 1 (
        echo [错误] 知识库向量入库失败，请检查 .env 中的 EMBEDDING_* 配置与网络连通性！
        pause
        exit /b 1
    )
) else (
    echo [3/4] 向量知识库 data\chroma_db 已就绪。
)

REM 5. 后台探测服务健康状态，就绪后自动打开浏览器
echo [4/4] 正在启动 FastAPI 服务 (http://127.0.0.1:8000) ...
echo       服务就绪后将自动打开浏览器；按 Ctrl+C 可停止服务。
echo.
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "for ($i=0; $i -lt 45; $i++) { try { $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 2; if ($r.StatusCode -eq 200) { Start-Process 'http://127.0.0.1:8000'; break } } catch {}; Start-Sleep -Milliseconds 1000 }"

uvicorn app.main:app --host 127.0.0.1 --port 8000

echo.
echo [提示] 服务已退出。
pause
endlocal
