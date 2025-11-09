"""
RL Learning Dashboard - Visualize DQN Learning Across Multiple Trials
Shows how the agent learns through experience, exploration, and Q-value updates
"""
import os
import sys
import json
import numpy as np
from pathlib import Path

import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.maze_env import MazeEnv

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "DQN Learning Progression"

# Global variables
env = None
episode_recordings = None  # Detailed recordings at key intervals
training_stats = None  # Overall training statistics

# Action colors
ACTION_COLORS = {
    0: '#FF6B6B',  # Red - UP
    1: '#4ECDC4',  # Teal - RIGHT
    2: '#95E1D3',  # Light teal - DOWN
    3: '#F38181',  # Pink - LEFT
}
ACTION_NAMES = {0: 'UP', 1: 'RIGHT', 2: 'DOWN', 3: 'LEFT'}

def load_data():
    """Load DQN training data and episode recordings."""
    global env, episode_recordings, training_stats

    env = MazeEnv()

    dqn_log_dir = Path('logs/dqn')
    if not dqn_log_dir.exists():
        print("[WARNING] No DQN logs found")
        return False

    # Load episode recordings
    recording_files = list(dqn_log_dir.glob('episode_recordings_*.json'))
    if not recording_files:
        print("[WARNING] No episode recordings found. Please train DQN with new code.")
        return False

    latest_recording_file = max(recording_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading episode recordings from: {latest_recording_file}")

    with open(latest_recording_file, 'r') as f:
        episode_recordings = json.load(f)

    # Load training stats
    stats_files = list(dqn_log_dir.glob('training_stats_*.json'))
    if stats_files:
        latest_stats_file = max(stats_files, key=lambda p: p.stat().st_mtime)
        with open(latest_stats_file, 'r') as f:
            training_stats = json.load(f)
        print(f"Loaded training stats from: {latest_stats_file}")

    print(f"[OK] Loaded {len(episode_recordings)} episode recordings")
    return True

def create_episode_comparison_viz(selected_episodes):
    """
    Compare multiple episodes side-by-side to show learning progression.
    """
    if not episode_recordings or not selected_episodes:
        return go.Figure().add_annotation(
            text="No episodes selected",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )

    maze = np.array(env.maze)
    num_episodes = len(selected_episodes)

    # Create subplots for each episode
    cols = min(3, num_episodes)
    rows = (num_episodes + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[f"Ep {ep} (ε={episode_recordings[ep]['epsilon']:.3f})"
                       for ep in selected_episodes],
        vertical_spacing=0.12 / rows,
        horizontal_spacing=0.08
    )

    for idx, ep_num in enumerate(selected_episodes):
        row = idx // cols + 1
        col = idx % cols + 1

        recording = episode_recordings[ep_num]
        positions = np.array(recording['positions'])

        # Add maze heatmap
        colorscale = [[0, 'white'], [0.25, 'black'], [0.5, 'gold'], [0.75, 'orange'], [1, 'purple']]
        fig.add_trace(
            go.Heatmap(z=maze, colorscale=colorscale, showscale=False, hoverinfo='skip'),
            row=row, col=col
        )

        # Add trajectory
        fig.add_trace(
            go.Scatter(
                x=positions[:, 1], y=positions[:, 0],
                mode='lines+markers',
                line=dict(color='cyan', width=2),
                marker=dict(size=4, color='cyan'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=row, col=col
        )

        # Mark start and end
        fig.add_trace(
            go.Scatter(x=[positions[0, 1]], y=[positions[0, 0]],
                      mode='markers', marker=dict(size=12, color='green', symbol='star'),
                      showlegend=False, hoverinfo='skip'),
            row=row, col=col
        )

        end_marker = 'star' if recording['reached_goal'] else 'x'
        end_color = 'gold' if recording['reached_goal'] else 'red'
        fig.add_trace(
            go.Scatter(x=[positions[-1, 1]], y=[positions[-1, 0]],
                      mode='markers', marker=dict(size=12, color=end_color, symbol=end_marker),
                      showlegend=False, hoverinfo='skip'),
            row=row, col=col
        )

        # Format axes
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=row, col=col)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, scaleanchor='x', row=row, col=col)

    fig.update_layout(
        title="Learning Progression - Episode Comparison",
        height=300 * rows,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig

def create_exploration_exploitation_viz(episode_num):
    """Show which actions were exploratory vs exploitative."""
    if not episode_recordings or episode_num not in episode_recordings:
        return go.Figure()

    recording = episode_recordings[episode_num]
    positions = np.array(recording['positions'])
    was_random = recording['action_was_random']

    fig = go.Figure()

    # Draw maze
    maze = np.array(env.maze)
    colorscale = [[0, 'white'], [0.25, 'black'], [0.5, 'gold'], [0.75, 'orange'], [1, 'purple']]
    fig.add_trace(go.Heatmap(z=maze, colorscale=colorscale, showscale=False, hoverinfo='skip'))

    # Separate exploration and exploitation steps
    exploit_positions = []
    explore_positions = []

    for i in range(len(was_random)):
        if was_random[i]:
            explore_positions.append(positions[i])
        else:
            exploit_positions.append(positions[i])

    # Plot exploitation steps (learned policy)
    if exploit_positions:
        exploit_positions = np.array(exploit_positions)
        fig.add_trace(go.Scatter(
            x=exploit_positions[:, 1], y=exploit_positions[:, 0],
            mode='markers',
            marker=dict(size=10, color='blue', symbol='circle'),
            name='Exploitation (Learned)'
        ))

    # Plot exploration steps (random)
    if explore_positions:
        explore_positions = np.array(explore_positions)
        fig.add_trace(go.Scatter(
            x=explore_positions[:, 1], y=explore_positions[:, 0],
            mode='markers',
            marker=dict(size=10, color='orange', symbol='x'),
            name='Exploration (Random)'
        ))

    # Mark start and goal
    fig.add_trace(go.Scatter(
        x=[positions[0, 1]], y=[positions[0, 0]],
        mode='markers', marker=dict(size=15, color='green', symbol='star'),
        name='Start', hoverinfo='skip'
    ))

    fig.update_layout(
        title=f"Episode {episode_num} - Exploration vs Exploitation (ε={recording['epsilon']:.3f})",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor='y'),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        hovermode='closest'
    )

    return fig

def create_q_value_evolution_viz():
    """Show how Q-values evolve at a specific state over training."""
    if not episode_recordings:
        return go.Figure()

    # Pick a key state (e.g., starting position [0,0])
    key_state_idx = 0  # First state of each episode

    episodes = sorted([int(k) for k in episode_recordings.keys()])
    q_values_over_time = {action: [] for action in range(4)}

    for ep in episodes:
        recording = episode_recordings[str(ep)]
        if len(recording['q_values']) > key_state_idx:
            q_vals = recording['q_values'][key_state_idx]
            for action in range(4):
                q_values_over_time[action].append(q_vals[action])

    fig = go.Figure()

    for action in range(4):
        fig.add_trace(go.Scatter(
            x=episodes[:len(q_values_over_time[action])],
            y=q_values_over_time[action],
            mode='lines+markers',
            name=f'{ACTION_NAMES[action]}',
            line=dict(color=ACTION_COLORS[action], width=2)
        ))

    fig.update_layout(
        title=f"Q-Value Evolution at Start State [0,0]",
        xaxis_title="Episode",
        yaxis_title="Q-Value",
        height=400,
        hovermode='x unified'
    )

    return fig

def create_learning_metrics_timeline():
    """Show overall learning metrics across all episodes."""
    if not training_stats:
        return go.Figure()

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Episode Reward',
            'Success Rate',
            'Epsilon Decay (Exploration)',
            'Average Q-Values',
            'Episode Length (Steps)',
            'Loss'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.12
    )

    episodes = training_stats['episode']

    # Reward with moving average
    fig.add_trace(
        go.Scatter(x=episodes, y=training_stats['reward'],
                  mode='lines', name='Reward', line=dict(color='lightblue', width=1),
                  opacity=0.5),
        row=1, col=1
    )
    window = 50
    if len(training_stats['reward']) >= window:
        moving_avg = np.convolve(training_stats['reward'], np.ones(window)/window, mode='valid')
        fig.add_trace(
            go.Scatter(x=list(range(window-1, len(episodes))), y=moving_avg,
                      mode='lines', name='50-ep MA', line=dict(color='blue', width=3)),
            row=1, col=1
        )

    # Success rate (moving average)
    fig.add_trace(
        go.Scatter(x=episodes, y=training_stats['success'],
                  mode='lines', name='Success', line=dict(color='green', width=2)),
        row=1, col=2
    )

    # Epsilon decay
    fig.add_trace(
        go.Scatter(x=episodes, y=training_stats['epsilon'],
                  mode='lines', name='Epsilon', line=dict(color='red', width=2),
                  fill='tozeroy'),
        row=2, col=1
    )

    # Q-values
    fig.add_trace(
        go.Scatter(x=episodes, y=training_stats['q_values'],
                  mode='lines', name='Avg Q-Value', line=dict(color='purple', width=2)),
        row=2, col=2
    )

    # Steps
    fig.add_trace(
        go.Scatter(x=episodes, y=training_stats['steps'],
                  mode='lines', name='Steps', line=dict(color='orange', width=2)),
        row=3, col=1
    )

    # Loss
    fig.add_trace(
        go.Scatter(x=episodes, y=training_stats['loss'],
                  mode='lines', name='Loss', line=dict(color='brown', width=2)),
        row=3, col=2
    )

    fig.update_xaxes(title_text="Episode")
    fig.update_layout(
        height=900,
        showlegend=False,
        margin=dict(l=40, r=20, t=80, b=40)
    )

    return fig

# App layout
app.layout = html.Div([
    html.H1("DQN Learning Progression - Trial-by-Trial Analysis",
            style={'textAlign': 'center', 'marginBottom': '10px'}),

    html.P("Watch how the agent improves through experience and Q-learning",
           style={'textAlign': 'center', 'color': 'gray', 'marginBottom': '20px'}),

    # Learning metrics timeline
    html.Div([
        html.H3("Overall Learning Metrics", style={'textAlign': 'center'}),
        dcc.Graph(id='learning-metrics-timeline'),
    ], style={'marginBottom': '30px', 'padding': '20px', 'backgroundColor': '#f9f9f9'}),

    # Q-value evolution
    html.Div([
        html.H3("Q-Value Evolution", style={'textAlign': 'center'}),
        html.P("See how action-values improve at the start state over training",
               style={'textAlign': 'center', 'color': 'gray'}),
        dcc.Graph(id='q-value-evolution'),
    ], style={'marginBottom': '30px', 'padding': '20px', 'backgroundColor': '#f0f8ff'}),

    # Episode comparison
    html.Div([
        html.Div([
            html.H3("Episode Inspector", style={'marginBottom': '20px'}),

            html.Label("Select Episodes to Compare:", style={'fontWeight': 'bold'}),
            dcc.Checklist(
                id='episode-selector',
                options=[],
                value=[],
                style={'marginBottom': '20px'}
            ),

            html.Label("Single Episode Details:", style={'fontWeight': 'bold', 'marginTop': '20px'}),
            dcc.Dropdown(
                id='single-episode-dropdown',
                options=[],
                value=None,
                placeholder="Select episode for detailed view"
            ),

            html.Div(id='episode-info', style={'marginTop': '20px', 'padding': '15px',
                                               'backgroundColor': '#e8f4f8', 'borderRadius': '5px'}),

        ], style={'width': '25%', 'display': 'inline-block', 'verticalAlign': 'top',
                 'padding': '20px', 'backgroundColor': '#fafafa'}),

        # Visualizations
        html.Div([
            dcc.Graph(id='episode-comparison-viz', style={'marginBottom': '20px'}),
            dcc.Graph(id='exploration-exploitation-viz'),
        ], style={'width': '75%', 'display': 'inline-block', 'padding': '20px'}),

    ]),
])

@callback(
    Output('episode-selector', 'options'),
    Output('episode-selector', 'value'),
    Output('single-episode-dropdown', 'options'),
    Output('learning-metrics-timeline', 'figure'),
    Output('q-value-evolution', 'figure'),
    Input('episode-selector', 'id')
)
def initialize_dashboard(_):
    """Initialize dashboard with loaded data."""
    if not episode_recordings:
        return [], [], [], go.Figure(), go.Figure()

    ep_nums = sorted([int(k) for k in episode_recordings.keys()])

    # Create options for episode selector
    options = [{'label': f'Episode {ep} (ε={episode_recordings[str(ep)]["epsilon"]:.3f})',
                'value': str(ep)} for ep in ep_nums]

    # Default selection: first 3 recorded episodes
    default_selection = [str(ep) for ep in ep_nums[:min(3, len(ep_nums))]]

    # Create figures
    metrics_fig = create_learning_metrics_timeline()
    q_value_fig = create_q_value_evolution_viz()

    return options, default_selection, options, metrics_fig, q_value_fig

@callback(
    Output('episode-comparison-viz', 'figure'),
    Input('episode-selector', 'value')
)
def update_comparison(selected_episodes):
    """Update episode comparison visualization."""
    if not selected_episodes:
        return go.Figure()
    return create_episode_comparison_viz(selected_episodes)

@callback(
    Output('exploration-exploitation-viz', 'figure'),
    Output('episode-info', 'children'),
    Input('single-episode-dropdown', 'value')
)
def update_single_episode_view(episode_num):
    """Update single episode detailed view."""
    if not episode_num or episode_num not in episode_recordings:
        return go.Figure(), html.P("Select an episode for details")

    recording = episode_recordings[episode_num]

    # Create info display
    random_actions = sum(recording['action_was_random'])
    total_actions = len(recording['action_was_random'])

    info = html.Div([
        html.H4(f"Episode {episode_num} Details"),
        html.P(f"Epsilon: {recording['epsilon']:.4f}"),
        html.P(f"Steps: {recording['steps']}"),
        html.P(f"Total Reward: {recording['total_reward']:.2f}"),
        html.P(f"Reached Goal: {'YES!' if recording['reached_goal'] else 'No'}"),
        html.P(f"Exploration: {random_actions}/{total_actions} actions ({random_actions/total_actions*100:.1f}%)"),
        html.P(f"Exploitation: {total_actions-random_actions}/{total_actions} actions ({(total_actions-random_actions)/total_actions*100:.1f}%)"),
    ])

    fig = create_exploration_exploitation_viz(episode_num)

    return fig, info

# Load data when module is imported
print("Loading DQN learning data...")
if load_data():
    print("[OK] Data loaded successfully!")
else:
    print("[ERROR] Failed to load data. Please train DQN with new code:")
    print("  python train_agents.py --dqn-only --quick")

if __name__ == '__main__':
    print("\nStarting RL Learning Dashboard...")
    print("\n" + "="*70)
    print("DQN Learning Progression Dashboard")
    print("="*70)
    print("\nFeatures:")
    print("  * Compare episodes across training to see improvement")
    print("  * Track exploration vs exploitation balance")
    print("  * Watch Q-values converge over time")
    print("  * Monitor learning metrics (reward, loss, epsilon)")
    print("\nDashboard will open at: http://127.0.0.1:8052")
    print("Press Ctrl+C to stop")
    print("="*70 + "\n")

    app.run(debug=False, host='127.0.0.1', port=8052)
