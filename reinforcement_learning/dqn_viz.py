from manim import *

class DQNFlow(Scene):
    def construct(self):
        title = Text("Deep Q-Network (DQN) Learning Flow", weight=BOLD).to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)

        # --- Nodes ---
        state = RoundedRectangle(corner_radius=0.2, width=3, height=1, color=BLUE)
        state_text = Text("State (sₜ)").move_to(state.get_center())
        state_group = VGroup(state, state_text).shift(3*LEFT)

        q_net = Rectangle(width=3.5, height=2.2, color=PURPLE)
        q_text = Text("Q-Network\nNeural Layers").move_to(q_net.get_center())
        q_group = VGroup(q_net, q_text)

        action = RoundedRectangle(corner_radius=0.2, width=3, height=1, color=GREEN)
        action_text = Text("Action (aₜ)").move_to(action.get_center())
        action_group = VGroup(action, action_text).shift(3*RIGHT)

        reward = RoundedRectangle(corner_radius=0.2, width=2.8, height=1, color=YELLOW)
        reward_text = Text("Reward (rₜ)").move_to(reward.get_center())
        reward_group = VGroup(reward, reward_text).next_to(action_group, DOWN, buff=1.2)

        update = RoundedRectangle(corner_radius=0.2, width=3.5, height=1, color=RED)
        update_text = Text("Q-Value Update\n(Loss Backpropagation)", t2c={"Loss": RED}).scale(0.8)
        update_text.move_to(update.get_center())
        update_group = VGroup(update, update_text).next_to(q_group, DOWN, buff=1.2)

        # --- Arrows ---
        arrows = [
            Arrow(state.get_right(), q_net.get_left(), buff=0.1, color=BLUE),
            Arrow(q_net.get_right(), action.get_left(), buff=0.1, color=GREEN),
            Arrow(action.get_bottom(), reward.get_top(), buff=0.1, color=YELLOW),
            Arrow(reward.get_left(), update.get_right(), buff=0.1, color=RED),
            CurvedArrow(update.get_top(), q_net.get_bottom(), radius=-2, color=RED)
        ]

        # --- Animation ---
        self.play(FadeIn(state_group))
        self.play(Create(arrows[0]), FadeIn(q_group))
        self.play(Create(arrows[1]), FadeIn(action_group))
        self.play(Create(arrows[2]), FadeIn(reward_group))
        self.play(Create(arrows[3]), FadeIn(update_group))
        self.play(Create(arrows[4]))
        self.wait(1)

        # --- Highlight loop meaning ---
        loop_text = Text("Experience Replay + Q-Value Update", color=RED).scale(0.7)
        loop_text.next_to(update_group, DOWN)
        self.play(Write(loop_text))
        self.wait(2)

        # --- Optional neuron activation illustration ---
        neurons = VGroup(*[
            Circle(radius=0.1, color=PURPLE, fill_opacity=0.7)
            for _ in range(15)
        ]).arrange_in_grid(rows=3, cols=5).move_to(q_group.get_center())

        self.play(FadeIn(neurons))
        self.play(Indicate(q_group, color=PURPLE), run_time=1.5)
        self.wait(1)
