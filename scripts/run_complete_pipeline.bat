@echo off
REM Complete Pipeline Automation Script for Windows
REM Usage: run_complete_pipeline.bat [quick] [gpu]

setlocal enabledelayedexpansion

REM Colors (limited in CMD)
set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

REM Configuration
set QUICK_MODE=false
set USE_GPU=false
set TIMESTAMP=%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set TIMESTAMP=%TIMESTAMP: =0%
set LOG_DIR=logs\pipeline_%TIMESTAMP%

REM Parse arguments
:parse_args
if "%~1"=="" goto end_parse
if /I "%~1"=="quick" set QUICK_MODE=true
if /I "%~1"=="gpu" set USE_GPU=true
if /I "%~1"=="--help" (
    echo Usage: %0 [quick] [gpu]
    echo   quick: Run with reduced epochs for testing
    echo   gpu: Use GPU acceleration for DQN
    exit /b 0
)
shift
goto parse_args
:end_parse

echo %BLUE%==========================================================================
echo MAZE NAVIGATION COMPARISON PIPELINE - WINDOWS
echo ==========================================================================%NC%
echo.
echo Configuration:
echo   Quick Mode: %QUICK_MODE%
echo   GPU Mode: %USE_GPU%
echo   Log Directory: %LOG_DIR%
echo.

REM Check Python
echo %BLUE%Checking Python installation...%NC%
python --version >nul 2>&1
if errorlevel 1 (
    echo %RED%Python not found. Please install Python 3.8 or higher.%NC%
    exit /b 1
)
echo %GREEN%Python found%NC%

REM Check dependencies
echo %BLUE%Checking dependencies...%NC%
python -c "import neat; import torch; import gymnasium" >nul 2>&1
if errorlevel 1 (
    echo %YELLOW%Installing dependencies...%NC%
    pip install -r requirements.txt
    if errorlevel 1 (
        echo %RED%Failed to install dependencies%NC%
        exit /b 1
    )
)
echo %GREEN%All dependencies installed%NC%
echo.

REM Create directories
echo %BLUE%Creating directories...%NC%
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "logs\neat" mkdir "logs\neat"
if not exist "logs\dqn" mkdir "logs\dqn"
if not exist "analysis" mkdir "analysis"
if not exist "assets\gifs" mkdir "assets\gifs"
if not exist "assets\images" mkdir "assets\images"
echo %GREEN%Directories created%NC%
echo.

REM Train NEAT
echo %BLUE%==========================================================================
echo TRAINING NEAT AGENT
echo ==========================================================================%NC%

if "%QUICK_MODE%"=="true" (
    set GENERATIONS=10
    echo Quick mode: Training for 10 generations
) else (
    set GENERATIONS=50
    echo Full mode: Training for 50 generations
)

cd neuroevolution
echo import sys > temp_train_neat.py
echo sys.path.append('..') >> temp_train_neat.py
echo from neat_solver import NEATMazeSolver, create_neat_config >> temp_train_neat.py
echo. >> temp_train_neat.py
echo print("Starting NEAT training...") >> temp_train_neat.py
echo config_path = create_neat_config('config-neat.txt') >> temp_train_neat.py
echo solver = NEATMazeSolver(config_path, log_dir='../logs/neat') >> temp_train_neat.py
echo winner = solver.train(generations=%GENERATIONS%) >> temp_train_neat.py
echo solver.visualize_training() >> temp_train_neat.py
echo solver.evaluate_best(render=False, num_episodes=10) >> temp_train_neat.py
echo print("NEAT training complete!") >> temp_train_neat.py

python temp_train_neat.py
if errorlevel 1 (
    echo %RED%NEAT training failed%NC%
    cd ..
    exit /b 1
)
del temp_train_neat.py
cd ..
echo %GREEN%NEAT training completed%NC%
echo.

REM Train DQN
echo %BLUE%==========================================================================
echo TRAINING DQN AGENT
echo ==========================================================================%NC%

if "%QUICK_MODE%"=="true" (
    set EPISODES=100
    echo Quick mode: Training for 100 episodes
) else (
    set EPISODES=500
    echo Full mode: Training for 500 episodes
)

cd reinforcement_learning
echo import sys > temp_train_dqn.py
echo sys.path.append('..') >> temp_train_dqn.py
echo from dqn_solver import DQNMazeSolver >> temp_train_dqn.py
echo from env.maze_env import MazeEnv >> temp_train_dqn.py
echo. >> temp_train_dqn.py
echo print("Starting DQN training...") >> temp_train_dqn.py
echo env = MazeEnv() >> temp_train_dqn.py
echo solver = DQNMazeSolver(env, log_dir='../logs/dqn') >> temp_train_dqn.py
echo solver.train(num_episodes=%EPISODES%, verbose=True) >> temp_train_dqn.py
echo solver.visualize_training() >> temp_train_dqn.py
echo solver.evaluate(num_episodes=10, render=False) >> temp_train_dqn.py
echo print("DQN training complete!") >> temp_train_dqn.py

python temp_train_dqn.py
if errorlevel 1 (
    echo %RED%DQN training failed%NC%
    cd ..
    exit /b 1
)
del temp_train_dqn.py
cd ..
echo %GREEN%DQN training completed%NC%
echo.

REM Generate comparisons
echo %BLUE%==========================================================================
echo GENERATING COMPARISON ANALYSIS
echo ==========================================================================%NC%

cd analysis
echo import sys > temp_compare.py
echo sys.path.append('..') >> temp_compare.py
echo from visualize_training import TrainingVisualizer >> temp_compare.py
echo. >> temp_compare.py
echo print("Creating comparison visualizations...") >> temp_compare.py
echo visualizer = TrainingVisualizer( >> temp_compare.py
echo     neat_log_dir='../logs/neat', >> temp_compare.py
echo     dqn_log_dir='../logs/dqn' >> temp_compare.py
echo ) >> temp_compare.py
echo. >> temp_compare.py
echo visualizer.create_comparison_dashboard() >> temp_compare.py
echo visualizer.visualize_decision_boundaries() >> temp_compare.py
echo visualizer.create_live_comparison() >> temp_compare.py
echo print("Generating adaptation animation...") >> temp_compare.py
echo visualizer.create_adaptation_animation(save_path='adaptation.gif') >> temp_compare.py
echo print("Comparison analysis complete!") >> temp_compare.py

python temp_compare.py
if errorlevel 1 (
    echo %RED%Comparison generation failed%NC%
    cd ..
    exit /b 1
)
del temp_compare.py
cd ..
echo %GREEN%Comparison analysis completed%NC%
echo.

REM Run robustness tests
echo %BLUE%==========================================================================
echo RUNNING ROBUSTNESS TESTS
echo ==========================================================================%NC%

cd analysis
echo import sys > temp_robustness.py
echo sys.path.append('..') >> temp_robustness.py
echo from robustness_tests import RobustnessTestSuite >> temp_robustness.py
echo. >> temp_robustness.py
echo print("Initializing robustness test suite...") >> temp_robustness.py
echo suite = RobustnessTestSuite( >> temp_robustness.py
echo     neat_model_path='../logs/neat/best_genome_gen_50.pkl', >> temp_robustness.py
echo     dqn_model_path='../logs/dqn/best_model.pth', >> temp_robustness.py
echo     neat_config_path='../neuroevolution/config-neat.txt' >> temp_robustness.py
echo ) >> temp_robustness.py
echo. >> temp_robustness.py
echo print("Running all robustness tests...") >> temp_robustness.py
echo suite.run_all_tests() >> temp_robustness.py
echo print("Robustness testing complete!") >> temp_robustness.py

python temp_robustness.py
if errorlevel 1 (
    echo %RED%Robustness testing failed%NC%
    cd ..
    exit /b 1
)
del temp_robustness.py
cd ..
echo %GREEN%Robustness testing completed%NC%
echo.

REM Generate report
echo %BLUE%Generating final report...%NC%
echo # Maze Navigation Comparison Report > "%LOG_DIR%\report.md"
echo Generated: %date% %time% >> "%LOG_DIR%\report.md"
echo. >> "%LOG_DIR%\report.md"
echo ## Configuration >> "%LOG_DIR%\report.md"
echo - Quick Mode: %QUICK_MODE% >> "%LOG_DIR%\report.md"
echo - GPU Enabled: %USE_GPU% >> "%LOG_DIR%\report.md"
echo - Timestamp: %TIMESTAMP% >> "%LOG_DIR%\report.md"
echo. >> "%LOG_DIR%\report.md"
echo ## Next Steps >> "%LOG_DIR%\report.md"
echo 1. Review visualizations in analysis\ directory >> "%LOG_DIR%\report.md"
echo 2. Examine detailed logs in logs\ directory >> "%LOG_DIR%\report.md"
echo 3. Open interactive dashboard: analysis\interactive_dashboard.html >> "%LOG_DIR%\report.md"
echo. >> "%LOG_DIR%\report.md"

REM Summary
echo.
echo %BLUE%==========================================================================
echo PIPELINE SUMMARY
echo ==========================================================================%NC%
echo.
echo %GREEN%Execution completed successfully!%NC%
echo.
echo Results Location: %LOG_DIR%
echo.
echo Key Outputs:
echo   - Visualizations: analysis\
echo   - Training Logs: logs\neat\ and logs\dqn\
echo   - Robustness Results: analysis\robustness_test_results.json
echo   - Summary Report: %LOG_DIR%\report.md
echo   - Interactive Dashboard: analysis\interactive_dashboard.html
echo.
echo To view results:
echo   1. cd analysis ^&^& dir *.png
echo   2. start analysis\interactive_dashboard.html
echo   3. type %LOG_DIR%\report.md
echo.
echo %GREEN%Pipeline completed successfully!%NC%
echo.

pause
