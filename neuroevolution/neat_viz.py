from manim import *
import random

class NEATEvolution(Scene):
    def construct(self):
        title = Text("NEAT: NeuroEvolution of Augmenting Topologies", weight=BOLD).to_edge(UP)
        self.play(Write(title))

        # --- Generation 1 population ---
        pop1 = self.create_population(label="Generation 1", y_shift=2)
        self.play(LaggedStart(*[FadeIn(p, shift=UP) for p in pop1], lag_ratio=0.1))
        self.wait(1)

        # --- Evaluation step ---
        eval_text = Text("Fitness Evaluation", color=YELLOW).scale(0.8).next_to(pop1, DOWN, buff=0.5)
        self.play(Write(eval_text))
        self.play(*[Indicate(p, color=YELLOW) for p in pop1])
        self.wait(0.5)

        # --- Selection: highlight top performers ---
        selected = random.sample(pop1, 2)
        self.play(*[p.animate.set_color(GREEN) for p in selected])
        self.play(Flash(selected[0]), Flash(selected[1]))
        sel_text = Text("Selection of Best Genomes", color=GREEN).scale(0.8).next_to(eval_text, DOWN)
        self.play(Write(sel_text))
        self.wait(0.5)

        # --- Crossover and mutation visual ---
        cross_text = Text("Crossover + Mutation", color=PURPLE).scale(0.8).next_to(sel_text, DOWN)
        self.play(Write(cross_text))
        self.wait(0.3)

        offspring = self.create_population(label="Offspring (Next Gen)", y_shift=-2)
        # connect parent to offspring
        lines = VGroup()
        for parent in selected:
            for child in offspring:
                line = DashedLine(parent.get_bottom(), child.get_top(), color=PURPLE, stroke_width=2)
                lines.add(line)
        self.play(Create(lines), LaggedStart(*[FadeIn(c, shift=DOWN) for c in offspring], lag_ratio=0.1))
        self.wait(1)

        # --- Mutation effect: change topology visually ---
        mutated_nodes = self.add_mutation_effect(offspring)
        self.play(*[Flash(m, color=RED) for m in mutated_nodes])
        mut_text = Text("Structural Mutation (Add Node/Connection)", color=RED).scale(0.8)
        mut_text.next_to(cross_text, DOWN)
        self.play(Write(mut_text))
        self.wait(1)

        # --- Transition to next generation label ---
        next_gen_label = Text("Generation 2", color=BLUE).next_to(offspring, DOWN, buff=0.4)
        self.play(Write(next_gen_label))
        self.wait(2)

    def create_population(self, label="Generation", y_shift=0):
        """Helper to create a group of small neural-net diagrams representing genomes."""
        population = VGroup()
        for i in range(5):
            net = self.create_mini_network()
            net.shift(RIGHT * (i - 2) * 1.8 + UP * y_shift)
            population.add(net)
        label_text = Text(label, color=WHITE).scale(0.7).next_to(population, UP)
        self.add(label_text)
        return population

    def create_mini_network(self):
        """Mini neural net diagram (3-layer) to represent a genome."""
        layer_sizes = [3, random.randint(2, 4), 2]
        layers = VGroup()
        for i, size in enumerate(layer_sizes):
            neurons = VGroup(*[Circle(radius=0.1, color=BLUE, fill_opacity=0.8) for _ in range(size)])
            neurons.arrange(DOWN, buff=0.15)
            neurons.shift(RIGHT * i * 0.5)
            layers.add(neurons)

        # connections
        conns = VGroup()
        for i in range(len(layer_sizes) - 1):
            for n1 in layers[i]:
                for n2 in layers[i + 1]:
                    if random.random() < 0.7:
                        conns.add(Line(n1.get_center(), n2.get_center(), stroke_width=1, color=GRAY))
        net = VGroup(conns, layers)
        return net

    def add_mutation_effect(self, offspring):
        """Add visual mutations to a subset of offspring networks."""
        mutated = []
        for net in offspring:
            if random.random() < 0.4:
                new_node = Circle(radius=0.1, color=RED, fill_opacity=0.9).move_to(net.get_center() + UP * 0.3)
                new_conn = Line(new_node.get_center(), random.choice(net[1][-1]).get_center(), color=RED, stroke_width=1.5)
                self.add(new_node, new_conn)
                mutated.append(new_node)
        return mutated
