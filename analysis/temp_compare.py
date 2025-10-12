import sys 
sys.path.append('..') 
from visualize_training import TrainingVisualizer 
 
print("Creating comparison visualizations...") 
visualizer = TrainingVisualizer( 
    neat_log_dir='../logs/neat', 
    dqn_log_dir='../logs/dqn' 
) 
 
visualizer.create_comparison_dashboard() 
visualizer.visualize_decision_boundaries() 
visualizer.create_live_comparison() 
print("Generating adaptation animation...") 
visualizer.create_adaptation_animation(save_path='adaptation.gif') 
print("Comparison analysis complete") 
