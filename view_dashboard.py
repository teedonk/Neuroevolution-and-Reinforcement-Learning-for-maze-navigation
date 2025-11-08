"""
Launch the interactive training dashboard with Plotly Dash.
Run: python view_dashboard.py
"""

import os
import sys
import webbrowser
import time
from threading import Timer

def main():
    print("="*70)
    print("INTERACTIVE TRAINING DASHBOARD")
    print("="*70)
    print()

    # Check if data files exist
    neat_file = os.path.join('analysis', 'neat_dashboard_data.json')
    dqn_file = os.path.join('analysis', 'dqn_dashboard_data.json')

    has_data = False
    if os.path.exists(neat_file):
        print("[OK] NEAT training data found")
        has_data = True
    else:
        print("[WARNING] NEAT data not found (will show DQN only)")

    if os.path.exists(dqn_file):
        print("[OK] DQN training data found")
        has_data = True
    else:
        print("[WARNING] DQN data not found (will show NEAT only)")

    if not has_data:
        print("\n[ERROR] No training data found!")
        print("   Please run training first: python train_agents.py --quick")
        return 1

    print("\nStarting interactive dashboard...")
    print("\nFeatures:")
    print("  * Click on maze cells to see Q-values (DQN)")
    print("  * Run live episodes with trained agents")
    print("  * Real-time training curves")
    print("  * Compare NEAT vs DQN performance")
    print("  * Interactive action heatmaps")
    print()

    # Auto-open browser after delay
    def open_browser():
        webbrowser.open('http://127.0.0.1:8050')

    Timer(1.5, open_browser).start()

    # Import and run dashboard
    try:
        from interactive_dashboard import app
        app.run(debug=False, host='127.0.0.1', port=8050)
    except Exception as e:
        print(f"\n[ERROR] Error starting dashboard: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())
