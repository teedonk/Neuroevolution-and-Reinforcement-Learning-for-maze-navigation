#!/bin/bash

###############################################################################
# Complete Pipeline Automation Script
# 
# This script runs the entire maze navigation comparison pipeline:
# 1. Environment setup
# 2. NEAT training
# 3. DQN training  
# 4. Comparison analysis
# 5. Robustness testing
# 6. Report generation
#
# Usage: bash run_complete_pipeline.sh [--quick] [--gpu]
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
QUICK_MODE=false
USE_GPU=false
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/pipeline_$TIMESTAMP"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --gpu)
            USE_GPU=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--quick] [--gpu]"
            echo "  --quick: Run with reduced epochs for testing"
            echo "  --gpu: Use GPU acceleration for DQN"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Functions
print_header() {
    echo -e "${BLUE}"
    echo "=========================================================================="
    echo "$1"
    echo "=========================================================================="
    echo -e "${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check dependencies
check_dependencies() {
    print_header "Checking Dependencies"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 not found. Please install Python 3.8 or higher."
        exit 1
    fi
    print_success "Python $(python3 --version) found"
    
    # Check required packages
    python3 -c "import neat; import torch; import gymnasium" 2>/dev/null
    if [ $? -eq 0 ]; then
        print_success "All required packages installed"
    else
        print_warning "Some packages missing. Installing..."
        pip install -r requirements.txt
    fi
    
    # Check GPU availability
    if [ "$USE_GPU" = true ]; then
        python3 -c "import torch; print('GPU available' if torch.cuda.is_available() else exit(1))" 2>/dev/null
        if [ $? -eq 0 ]; then
            print_success "GPU acceleration available"
        else
            print_warning "GPU not available. Falling back to CPU."
            USE_GPU=false
        fi
    fi
    
    echo ""
}

# Setup directories
setup_directories() {
    print_header "Setting Up Directories"
    
    mkdir -p "$LOG_DIR"
    mkdir -p logs/neat
    mkdir -p logs/dqn
    mkdir -p analysis
    mkdir -p assets/gifs
    mkdir -p assets/images
    
    print_success "Directories created"
    echo ""
}

# Train NEAT
train_neat() {
    print_header "Training NEAT Agent"
    
    if [ "$QUICK_MODE" = true ]; then
        GENERATIONS=10
        print_info "Quick mode: Training for $GENERATIONS generations"
    else
        GENERATIONS=50
        print_info "Full mode: Training for $GENERATIONS generations"
    fi
    
    cd neuroevolution
    
    cat > temp_train_neat.py << EOF
import sys
sys.path.append('..')
from neat_solver import NEATMazeSolver, create_neat_config

print("Starting NEAT training...")
config_path = create_neat_config('config-neat.txt')
solver = NEATMazeSolver(config_path, log_dir='../logs/neat')
winner = solver.train(generations=$GENERATIONS)
solver.visualize_training()
solver.evaluate_best(render=False, num_episodes=10)
print("NEAT training complete!")
EOF
    
    python3 temp_train_neat.py
    rm temp_train_neat.py
    
    cd ..
    print_success "NEAT training completed"
    echo ""
}

# Train DQN
train_dqn() {
    print_header "Training DQN Agent"
    
    if [ "$QUICK_MODE" = true ]; then
        EPISODES=100
        print_info "Quick mode: Training for $EPISODES episodes"
    else
        EPISODES=500
        print_info "Full mode: Training for $EPISODES episodes"
    fi
    
    cd reinforcement_learning
    
    cat > temp_train_dqn.py << EOF
import sys
sys.path.append('..')
from dqn_solver import DQNMazeSolver
from env.maze_env import MazeEnv

print("Starting DQN training...")
env = MazeEnv()
solver = DQNMazeSolver(env, log_dir='../logs/dqn')
solver.train(num_episodes=$EPISODES, verbose=True)
solver.visualize_training()
solver.evaluate(num_episodes=10, render=False)
print("DQN training complete!")
EOF
    
    python3 temp_train_dqn.py
    rm temp_train_dqn.py
    
    cd ..
    print_success "DQN training completed"
    echo ""
}

# Generate comparisons
generate_comparisons() {
    print_header "Generating Comparison Analysis"
    
    cd analysis
    
    cat > temp_compare.py << EOF
import sys
sys.path.append('..')
from visualize_training import TrainingVisualizer

print("Creating comparison visualizations...")
visualizer = TrainingVisualizer(
    neat_log_dir='../logs/neat',
    dqn_log_dir='../logs/dqn'
)

# Create all visualizations
visualizer.create_comparison_dashboard()
visualizer.visualize_decision_boundaries()
visualizer.create_live_comparison()

print("Generating adaptation animation...")
visualizer.create_adaptation_animation(save_path='adaptation.gif')

print("Comparison analysis complete!")
EOF
    
    python3 temp_compare.py
    rm temp_compare.py
    
    cd ..
    print_success "Comparison analysis completed"
    echo ""
}

# Run robustness tests
run_robustness_tests() {
    print_header "Running Robustness Tests"
    
    cd analysis
    
    cat > temp_robustness.py << EOF
import sys
sys.path.append('..')
from robustness_tests import RobustnessTestSuite

print("Initializing robustness test suite...")
suite = RobustnessTestSuite(
    neat_model_path='../logs/neat/best_genome_gen_50.pkl',
    dqn_model_path='../logs/dqn/best_model.pth',
    neat_config_path='../neuroevolution/config-neat.txt'
)

print("Running all robustness tests...")
suite.run_all_tests()

print("Robustness testing complete!")
EOF
    
    python3 temp_robustness.py
    rm temp_robustness.py
    
    cd ..
    print_success "Robustness testing completed"
    echo ""
}

# Generate final report
generate_report() {
    print_header "Generating Final Report"
    
    cat > "$LOG_DIR/report.md" << EOF
# Maze Navigation Comparison Report
Generated: $(date)

## Configuration
- Quick Mode: $QUICK_MODE
- GPU Enabled: $USE_GPU
- Timestamp: $TIMESTAMP

## Training Summary

### NEAT
- Training completed: ✅
- Generations: $([ "$QUICK_MODE" = true ] && echo "10" || echo "50")
- Logs: logs/neat/

### DQN
- Training completed: ✅
- Episodes: $([ "$QUICK_MODE" = true ] && echo "100" || echo "500")
- Logs: logs/dqn/

## Generated Artifacts

### Visualizations
- ✅ comparison_dashboard.png
- ✅ decision_boundaries.png
- ✅ live_comparison.png
- ✅ adaptation.gif
- ✅ training_curves.png

### Analysis Results
- ✅ noise_sensitivity.png
- ✅ generalization.png
- ✅ failure_modes.png
- ✅ robustness_scores.png

### Logs
- ✅ NEAT training logs
- ✅ DQN training logs
- ✅ Robustness test results

## Next Steps

1. Review visualizations in analysis/ directory
2. Examine detailed logs in logs/ directory
3. Open interactive dashboard: analysis/interactive_dashboard.html
4. Check robustness test results: analysis/robustness_test_results.json

## Performance Summary

See individual log files for detailed metrics:
- NEAT: logs/neat/final_stats_*.json
- DQN: logs/dqn/training_stats_*.json

## Citation

If you use these results, please cite:
\`\`\`
@software{Neuroevolution-and-Reinforcement-Learning-for-maze-navigation},
  title = {Maze Navigation: NEAT vs DQN Comparison},
  date = {$(date +"25-10-12")},
  pipeline_id = {$TIMESTAMP}
}
\`\`\`

---
Generated by automated pipeline
EOF
    
    print_success "Report generated: $LOG_DIR/report.md"
    echo ""
}

# Create summary
create_summary() {
    print_header "Pipeline Summary"
    
    echo ""
    echo "Execution completed successfully!"
    echo ""
    echo "Results Location: $LOG_DIR"
    echo ""
    echo "Key Outputs:"
    echo "  📊 Visualizations: analysis/"
    echo "  📈 Training Logs: logs/neat/ and logs/dqn/"
    echo "  🔬 Robustness Results: analysis/robustness_test_results.json"
    echo "  📝 Summary Report: $LOG_DIR/report.md"
    echo "  🌐 Interactive Dashboard: analysis/interactive_dashboard.html"
    echo ""
    echo "To view results:"
    echo "  1. cd analysis && open *.png"
    echo "  2. open analysis/interactive_dashboard.html"
    echo "  3. cat $LOG_DIR/report.md"
    echo ""
    
    # Calculate total time
    DURATION=$((SECONDS / 60))
    echo "Total execution time: ${DURATION} minutes"
    echo ""
}

# Error handling
handle_error() {
    print_error "Pipeline failed at step: $1"
    print_info "Check logs in $LOG_DIR for details"
    exit 1
}

###############################################################################
# Main Pipeline Execution
###############################################################################

main() {
    # Start timer
    SECONDS=0
    
    print_header "🧠 Maze Navigation Comparison Pipeline"
    echo ""
    echo "Configuration:"
    echo "  Quick Mode: $QUICK_MODE"
    echo "  GPU Mode: $USE_GPU"
    echo "  Log Directory: $LOG_DIR"
    echo ""
    
    # Execute pipeline steps
    check_dependencies || handle_error "Dependency check"
    setup_directories || handle_error "Directory setup"
    
    print_info "Starting training phase..."
    train_neat || handle_error "NEAT training"
    train_dqn || handle_error "DQN training"
    
    print_info "Starting analysis phase..."
    generate_comparisons || handle_error "Comparison generation"
    run_robustness_tests || handle_error "Robustness testing"
    
    print_info "Generating reports..."
    generate_report || handle_error "Report generation"
    
    create_summary
    
    print_success "🎉 Pipeline completed successfully!"
}

# Trap errors
trap 'handle_error "Unknown error"' ERR

# Run main
main "$@"
