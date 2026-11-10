@echo off
rem One-time setup for this laptop. Run from this folder: double-click, or "setup_windows.bat" in a terminal.
cd /d "%~dp0"
python --version || (echo Python not found. Open "Anaconda Prompt", cd to this folder and run this file again. & pause & exit /b 1)
python -m venv .venv || (pause & exit /b 1)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu || (pause & exit /b 1)
pip install -e ".[dev]" pyarrow pandas || (pause & exit /b 1)
python -m spacy download en_core_web_sm || (pause & exit /b 1)
set PYTHONPATH=.
echo.
echo ===== Fast tests (expect: 129 passed, 19 deselected) =====
pytest tests/ -m "not integration" -q -p no:cacheprovider
echo.
echo ===== Internal benchmark with real models (expect: Accuracy 0.75) =====
python benchmarks/mcpevol_runner.py --dataset benchmarks/mcpevol.json --output benchmarks/results_local.json
echo.
echo Setup finished. To work later: open a terminal here, run  .venv\Scripts\activate  then  set PYTHONPATH=.
pause
