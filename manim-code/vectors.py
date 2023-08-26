from manim import *

class VectorAnimation(Scene):
    def construct(self):
        # Starting with a single point in space
        point = Dot(ORIGIN)
        self.add(point)
        self.wait(1)

        # Draw an arrow extending from the point in any direction
        vector = Arrow(ORIGIN, UP+RIGHT, buff=0)
        self.play(GrowArrow(vector))
        self.wait(1)

        # Display labels
        vector_label = MathTex(r"\vec{a}", color=BLUE).next_to(vector, RIGHT)
        self.play(Write(vector_label))
        self.wait(1)

        # Add an explanation
        explanation = MathTex(r"Here, the vector is described by its length (r) and angle (", r"\theta", r").")
        explanation.to_edge(UP)
        self.play(Write(explanation))
        self.wait(2)

        # Draw a second vector
        vector2 = Arrow(vector.get_end(), vector.get_end() + 2*LEFT, buff=0, color=YELLOW)
        self.play(GrowArrow(vector2))
        self.wait(1)

        # Display label for the second vector
        vector2_label = MathTex(r"\vec{b}", color=YELLOW).next_to(vector2, LEFT)
        self.play(Write(vector2_label))
        self.wait(1)

        # Add the vectors
        sum_vector = Arrow(ORIGIN, vector.get_end() + vector2.get_end(), color=GREEN).next_to(vector2, LEFT)
        self.play(TransformFromCopy(VGroup(vector, vector2), sum_vector))
        self.wait(1)

        # Display label for the resultant vector
        sum_vector_label = MathTex(r"\vec{a} + \vec{b}", color=GREEN).next_to(sum_vector, LEFT)
        self.play(Write(sum_vector_label))
        self.wait(2)