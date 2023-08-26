from manim import *
import numpy as np

class VectorAdditionScene(Scene):
    def construct(self):
        plane = PolarPlane(
            azimuth_units="PI radians",
            size=8,  # increased size
            azimuth_label_font_size=0.7,
            radius_config={"font_size": 0.7},
        ).add_coordinates().set_opacity(0.7)

        self.vec_a_end = ValueTracker(PI/4)
        self.vec_a_length = ValueTracker(1.5)  # length of vector a

        # Create the initial vector
        vec_a = always_redraw(lambda: Arrow(plane.c2p(0, 0), plane.polar_to_point(self.vec_a_length.get_value(), self.vec_a_end.get_value())))
        vec_a_label = always_redraw(lambda: MathTex(r"\vec{a} = (r, \theta)").next_to(vec_a.get_end(), UP).scale(.5))

        # Initial scene setup
        self.play(FadeIn(plane))
        self.play(GrowArrow(vec_a), Write(vec_a_label))
        self.wait(1)

        # Animate vector moving
        self.play(self.vec_a_end.animate.set_value(PI/2), self.vec_a_length.animate.set_value(3))  # animate length and direction
        self.wait(1)

        # Create the second vector
        vec_b = always_redraw(lambda: Arrow(vec_a.get_end(), vec_a.get_end() + plane.polar_to_point(2, PI/6), color=YELLOW))  # increased length
        vec_b_label = always_redraw(lambda: MathTex(r"\vec{b} = (r, \theta)").next_to(vec_b.get_end(), RIGHT, buff=0.4).set_color(YELLOW).scale(.5))

        self.play(GrowArrow(vec_b), Write(vec_b_label))
        self.wait(1)

        # Create the sum vector
        vec_c = always_redraw(lambda: Arrow(plane.c2p(0, 0), vec_b.get_end(), color=GREEN))
        vec_c_label = always_redraw(lambda: MathTex(r"\vec{c} = \vec{a} + \vec{b}").next_to(vec_c.get_center(), RIGHT).set_color(GREEN).scale(.5))

        self.play(GrowArrow(vec_c), Write(vec_c_label))
        self.wait(1)
        
class VectorConcept(Scene):
    def construct(self):
        theta=ValueTracker(PI/4)
        plane=NumberPlane()

        vect_1=always_redraw(lambda: Arrow(plane.c2p(0, 0), plane.c2p(4.25*np.cos(theta.get_value()), 
                                                                      4.25*np.sin(theta.get_value())), 
                                           buff=0, color=RED))

        dot_orig=Dot(ORIGIN, color=BLUE)
        dot_tip=Dot(plane.c2p(3,3), color=YELLOW, radius=0.1)
        origin_text = MathTex(r'O\\(0, 0)').next_to(dot_orig, DOWN, buff=0.1)
        tip_tex=MathTex("A", color=YELLOW).next_to(dot_tip, UP+LEFT)

        label_group=VGroup(tip_tex, origin_text)
        dot_group=VGroup(dot_tip, dot_orig)
        angle=Angle(plane.get_x_axis(), vect_1, radius=1, color=WHITE)
         
        self.play(DrawBorderThenFill(plane))
        #self.wait(2)
        self.play(LaggedStart(FadeIn(dot_tip),Write(tip_tex), FocusOn(dot_tip, color=YELLOW),
                              lag_ratio=0.15), run_time=4)
         
        self.play(LaggedStart(FadeIn(dot_orig),Write(origin_text), FocusOn(dot_orig, color=BLUE),
                              lag_ratio=0.15), run_time=4)
        #self.wait()
        vec=self.play(GrowArrow(vect_1))
        self.wait()
        self.play(theta.animate.set_value(9*PI/4), run_time=4)
        self.wait()
        self.add(angle)
        self.wait()

class FourierSeries(Scene):
    def construct(self):
        # number of components
        n_components = 4

        # frequencies of the component waves
        frequencies = [2 * np.pi * (i+1) for i in range(n_components)]

        # amplitudes of the component waves
        amplitudes = [1/(i+1)/3 for i in range(n_components)]

        # create the complex wave as a sum of the component waves
        x_vals = np.linspace(-1.5*np.pi, 1.5*np.pi, 1000) # Reduce the range of x vals
        y_vals = sum(amplitudes[i] * np.sin(frequencies[i] * x_vals) for i in range(n_components))
        complex_wave = np.vstack((x_vals, y_vals, np.zeros_like(x_vals))).T
        complex_wave_path = VMobject().set_points_as_corners(complex_wave)

        # display the complex wave
        self.play(Create(complex_wave_path))
        self.wait(2)

        # move the complex wave to the top of the screen
        self.play(ApplyMethod(complex_wave_path.shift, 2*UP))
        self.wait(2)

        # display each component wave
        component_waves = []
        for i in range(n_components):
            y_vals = amplitudes[i] * np.sin(frequencies[i] * x_vals)
            wave = np.vstack((x_vals, y_vals, np.zeros_like(x_vals))).T
            wave_path = VMobject().set_points_as_corners(wave).shift(2*DOWN*(i+1)/n_components)
            component_waves.append(wave_path)

            # create a lab for the wave
            lab = Text(f"n={i+1}").next_to(wave_path, LEFT).scale(0.5)

            # animate the wave and the lab
            self.play(Create(wave_path), Write(lab))
            self.wait(1)

        # combine the component waves into the original complex wave
        for wave in component_waves:
            self.play(Transform(wave, complex_wave_path))
            self.wait(1)

        # remove all the component waves
        for wave in component_waves:
            self.remove(wave)

        self.wait(2)
        

from manim import *
import numpy as np

class FourierSeriesSum(Scene):
    def construct(self):
        # number of components
        n_components = 4

        # frequencies of the component waves
        frequencies = [2 * np.pi * (i+1) for i in range(n_components)]

        # amplitudes of the component waves
        amplitudes = [1/(i+1)/3 for i in range(n_components)]

        # create the complex wave as a sum of the component waves
        x_vals = np.linspace(-1.5*np.pi, 1.5*np.pi, 1000)
        y_vals = sum(amplitudes[i] * np.sin(frequencies[i] * x_vals) for i in range(n_components))
        complex_wave = np.vstack((x_vals, y_vals, np.zeros_like(x_vals))).T
        complex_wave_path = VMobject().set_points_as_corners(complex_wave)

        # display the complex wave
        self.play(Create(complex_wave_path))
        self.wait(2)

        # move the complex wave to the top of the screen
        self.play(ApplyMethod(complex_wave_path.shift, 2*UP))
        self.wait(2)

        # display each component wave
        component_waves = []
        labels = []
        for i in range(n_components):
            y_vals = amplitudes[i] * np.sin(frequencies[i] * x_vals)
            wave = np.vstack((x_vals, y_vals, np.zeros_like(x_vals))).T
            wave_path = VMobject().set_points_as_corners(wave).shift(2*DOWN*(i+1)/n_components)
            component_waves.append(wave_path)

            # create a label for the wave
            label = Text(f"n={i+1}").next_to(wave_path, LEFT).scale(.5)
            labels.append(label)

            # animate the wave and the label
            self.play(Create(wave_path), Write(label))
            self.wait(1)

        # merge component waves into the sum of itself and the following waves
        for i in reversed(range(n_components-1)):
            y_vals = sum(amplitudes[j] * np.sin(frequencies[j] * x_vals) for j in range(i, n_components))
            wave = np.vstack((x_vals, y_vals, np.zeros_like(x_vals))).T
            sum_wave_path = VMobject().set_points_as_corners(wave).shift(2*DOWN*(i+1)/n_components)
            
            self.play(Transform(component_waves[i], sum_wave_path), ApplyMethod(component_waves[i+1].shift, 2*UP/n_components), FadeOut(component_waves[i+1], shift=UP/2), FadeOut(labels[i+1]))
            labels[i].become(Text(f"n= {i+1} + {i+2}")).next_to(component_waves[i], LEFT).scale(.5)
            self.wait(1)

        # remove all the component waves
        for wave in component_waves:
            self.remove(wave)

        self.wait(2)
        
class SinCosSum(MovingCameraScene):
    def construct(self):
        # defining x vals
        x_vals = np.linspace(-2 * np.pi, 2 * np.pi, 1000)

        # defining y vals for sin and cos waves
        sin_y_vals = np.sin(x_vals)
        cos_y_vals = np.cos(x_vals)

        # creating sin and cos waves
        sin_wave = np.vstack((x_vals, sin_y_vals, np.zeros_like(x_vals))).T
        cos_wave = np.vstack((x_vals, cos_y_vals, np.zeros_like(x_vals))).T

        # creating sin and cos wave paths
        sin_wave_path = VMobject().set_points_as_corners(sin_wave).shift(2*UP)
        cos_wave_path = VMobject().set_points_as_corners(cos_wave).shift(2*DOWN)

        # adding sin and cos waves to the scene
        self.play(Create(sin_wave_path), Create(cos_wave_path))
        self.wait(1)

        # custom function to convert number to radian label
        def number_to_radian(d):
            fraction = d / np.pi
            if fraction == 0:
                return MathTex("0")
            elif fraction == 1:
                return MathTex("\\pi")
            elif fraction == -1:
                return MathTex("-\\pi")
            elif int(fraction) == fraction:
                return MathTex(f"{int(fraction)}\\pi")
            else:
                return MathTex(f"\\frac{{{int(fraction * 2)}}}{{2}}\\pi")

        # creating number line for sin and cos wave
        sin_wave_number_line = NumberLine(
            x_range=[-2 * np.pi, 2 * np.pi, np.pi / 2], 
            include_numbers=False, 
            font_size=20, 
            include_tip=True, 
            numbers_to_exclude=[0],
            label_direction=DOWN
        ).shift(2*UP)

        cos_wave_number_line = NumberLine(
            x_range=[-2 * np.pi, 2 * np.pi, np.pi / 2], 
            include_numbers=False, 
            font_size=20, 
            include_tip=True, 
            numbers_to_exclude=[0],
            label_direction=UP
        ).shift(2*DOWN)

        # manually add number labels to number lines
        for number in np.arange(-2*np.pi, 2*np.pi+np.pi/2, np.pi/2):
            sin_wave_number_line.add(number_to_radian(number).next_to(sin_wave_number_line.n2p(number), DOWN).scale(.5))
            cos_wave_number_line.add(number_to_radian(number).next_to(cos_wave_number_line.n2p(number), DOWN).scale(.5))

        # adding number lines to the scene
        self.play(Create(sin_wave_number_line), Create(cos_wave_number_line))
        self.wait(1)

        # create vertical dashed lines for every pi/2
        for x in np.arange(-2*np.pi, 2*np.pi, np.pi/2):
            self.play(Create(DashedLine(start=[x, 3, 0], end=[x, -3, 0]).set_stroke(width=0.5)))
        self.wait(1)

        # zooming in on pi/2
        self.play(
            self.camera.frame.animate.scale(0.7).move_to([0, 0, 0])
        )
        self.wait(2)

        # creating boxes at every pi/2
        b1 = Rectangle(width=0.2, height=0.2, color=BLUE_A).move_to([0, np.sin(0) + 2, 0])
        b2 = Rectangle(width=0.2, height=0.2, color=BLUE).move_to([0, np.cos(0) - 2, 0])
        self.play(
            Create(b1),
            Create(b2),
        )
        self.wait(2)

        # fade out unwanted components
        self.play(
            FadeOut(sin_wave_number_line),
            FadeOut(cos_wave_number_line),
            FadeOut(b1),
            FadeOut(b2),
        )
        
        self.wait(2)
        self.play(
            self.camera.frame.animate.scale(1.5)
        )

        # add equation text
        equation = MathTex("sin(0) + cos(0) = ?").scale(0.7).to_edge(UP)
        self.play(Write(equation))
        self.wait(2)
        
        sum_vals = []
        for i in range(len(sin_y_vals)):
            sum_vals.append(sin_y_vals[i] + cos_y_vals[i])
            

        # merge waves
        sin_cos_wave = np.vstack((x_vals, sum_vals, np.zeros_like(x_vals))).T
        sin_cos_wave_path = VMobject().set_points_as_corners(sin_cos_wave)

        self.play(
            Transform(sin_wave_path, sin_cos_wave_path),
            Transform(cos_wave_path, sin_cos_wave_path),
            run_time=2
        )
        self.wait(2)

        # draw number line through the sum wave
        sum_wave_number_line = Axes(
            x_range=[-2 * np.pi, 2 * np.pi, np.pi / 2], 
            y_range=[-2, 2, .5],
        )
        self.play(Create(sum_wave_number_line))
        self.wait(2)

        # change "?" to "1" in the equation
        new_equation = MathTex("sin(0) + cos(0) = 1").scale(0.7).to_edge(UP)
        self.play(Transform(equation, new_equation))
        self.wait(2)
        
class Epicycle(Scene):
    def construct(self) -> None:
        m1=self.vecs(mu=2.2)
        m2=self.vecs(ran=range(3,20,2),mu=0)
        m3 = self.vecs(ran=range(3, 250, 2), mu=-2.2)
        label=Tex(f"n={len(range(3,10,2))+1}",f"n={len(range(3,20,2))+1}",f"n={len(range(3,250,2))+1}")
        label[0].next_to(m1,LEFT)
        label[1].next_to(m2, LEFT)
        label[2].next_to(m3, LEFT)
        self.add(m1,m2,m3,label)
        self.wait(20)

    def vecs(self,ran=range(3,10,2),mu=2):
        vect = Vector(color=GREY_A)
        val = ValueTracker(0)
        rate = 0.24
        val.add_updater(lambda v, dt: v.set_value(dt * rate))
        self.add(val)
        vect.add_updater(lambda v: v.set_angle(v.get_angle() + 1 * TAU * val.get_value()))

        def update(pv, f):

            def updat(mob):
                mob.set_angle(mob.get_angle() + f * val.get_value() * TAU, about_point=pv.get_end())
                mob.shift(pv.get_end() - mob.get_start())

            return updat

        last = vect
        vgroup = VGroup()
        rang = ran
        for i in rang:
            vect2 = Vector().scale(1 / i)
            vect2.add_updater(update(last, i))
            vgroup.add(vect2)
            last = vect2

        line = Line(vgroup[-1].get_end(), vgroup[-1].get_end() + np.array([1 - vgroup[-1].get_end()[0] + 1, 0, 0]))
        line.add_updater(
            lambda v: v.become(
                Line(vgroup[-1].get_end(), vgroup[-1].get_end() + np.array([1 - vgroup[-1].get_end()[0] + 1, 0, 0]))))

        path = Line(vgroup[-1].get_end(), vgroup[-1].get_end() + UP * 0.00001).set_stroke(width=0.5)

        circ3 = Circle()

        def cupdat(v):
            def upd(mob):
                mob.move_to(v.move_to(v.get_start()))

            return upd

        for i, j, c in zip(range(len(rang)), rang,
                           np.random.choice([YELLOW, BLUE, GREEN, PINK, PURPLE, MAROON, GOLD, TEAL], size=len(rang))):
            circ = Circle(radius=1 / j, color=c).set_stroke(width=2)

            circ.add_updater(cupdat(vgroup[i]))
            self.add(circ)

        def path_update(mob):
            line1 = Line(path.get_end(), line.get_end())
            path.append_vectorized_mobject(line1)
            path.shift(RIGHT * val.get_value() * 3)
            mob.become(path)

        path.add_updater(path_update)

        fgroup = VGroup(vect, vgroup, line, path, circ3)
        fgroup.shift( mu*UP)
        return fgroup

class frequencyDomain(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes() 
        x_label = Text("Time").next_to(np.array([2*PI, 0, 0]), DOWN).scale(0.5)
        y_label = Text("Amplitude").next_to(np.array([0, 1, 0]), UL).scale(0.5)
        z_label = Text("Frequency").next_to(np.array([0, 0, 3]), LEFT*12).scale(0.5)

       
        self.set_camera_orientation(phi=180 * DEGREES, theta=-90 * DEGREES, gamma=0*DEGREES)

        self.play(Create(axes))
        self.wait()
        
        waves = [
            ParametricFunction(
                self.create_wave_func(freq), color=np.random.choice([YELLOW, BLUE, GREEN, PINK, PURPLE, MAROON, GOLD, TEAL, BLUE_B, YELLOW_B]), t_range=[-2 * PI, 2 * PI]
            )
            for freq in range(-2, 3)
        ]

        lines = [
            Line3D( 
                start=np.array([0, -1, freq]), 
                end=np.array([0, 1, freq]), 
                color=BLUE
            )
            for freq in [-2, -1, .5, 1, 2]
        ]

        for wave in waves:
            self.play(Create(wave))

        # Reveal the Z axis by rotating the scene
        self.move_camera(phi=90 * DEGREES, theta=0 * DEGREES, gamma=-90*DEGREES, run_time=3)
        for wave, line in zip(waves, lines):
            self.play(FadeOut(wave), FadeIn(line))

        self.wait()

    def create_wave_func(self, freq):
        if freq == 0:
            freq = .5
        def func(t):
            return np.array([
                t,
                np.sin(freq * t),  # amplitude
                freq,  # frequency
            ])
        return func

class EulersFormulaScene(Scene):
    def construct(self):
        # Create complex plane
        plane = ComplexPlane().add_coordinates()
        plane.y_axis.label_direction = DOWN
        plane.x_axis.label_direction = DOWN

        # Change labels of axes
        plane.x_axis.add_numbers([i for i in range(-5, 6)])
        plane.y_axis.add_numbers([i for i in range(-5, 6)], direction=LEFT)

        x_label = MathTex("R").next_to(plane, RIGHT, buff=0.5)
        y_label = MathTex("i").next_to(plane, UP, buff=0.5)
        
        # Circle on the plane
        circle = Circle(radius=2)
        circle.move_to(plane.c2p(0, 0))
        
        # Arrow
        arrow = Arrow(start=plane.c2p(0, 0), end=plane.c2p(2, 0), buff=0)
        
        # Add Euler's formula equation
        theta = ValueTracker(0)
        equation = MathTex(
            "\\cos(", "0.00", ") + i\\sin(", "0.00", ") = e^{i", "0.00", "}"
        ).to_corner(UR)

        def update_equation():
            for i in [1, 3, 5]:
                equation[i].become(MathTex(f"{theta.get_value():.2f}"))
                equation[i].next_to(equation[i-1], RIGHT/2.5)

        equation.add_updater(lambda m: update_equation())
        
        self.add(plane, circle, x_label, y_label, arrow, equation)

        # Animate the arrow
        rotation_speed = TAU / 8  # One full rotation every 8 seconds
        self.total_time = 0

        def update_arrow(obj, dt):
            self.total_time += dt
            angle = self.total_time * rotation_speed
            obj.become(Arrow(start=plane.c2p(0, 0), end=plane.polar_to_point(2, angle), buff=0))
            theta.set_value(angle)

        arrow.add_updater(update_arrow)
        self.wait(16)  # Show 2 full rotations
        arrow.clear_updaters()
        equation.clear_updaters()
        
from typing import Tuple, List, Union

class ThreeDSpiral(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(
            x_range=(-0.1, 4.25),
            y_range=(-1.5, 1.5),
            z_range=(-1.5, 1.5),
            y_length=5, z_length=5,
        )

        camera_orig_phi, camera_orig_theta = 75*DEGREES, -30*DEGREES

        curve, curve_extension, formula = self.show_curve(
            axes=axes, camera_orig_phi=camera_orig_phi, camera_orig_theta=camera_orig_theta)
        self.wait()

        self.show_sin(curve=curve, formula=formula,
                      camera_orig_phi=camera_orig_phi, camera_orig_theta=camera_orig_theta)
        self.wait()

        self.show_cos(curve=curve, formula=formula,
                      camera_orig_phi=camera_orig_phi, camera_orig_theta=camera_orig_theta)
        self.wait()

        self.play(FadeOut(axes, curve, curve_extension, formula, shift=IN))
        self.wait()

    def show_curve(self, axes, camera_orig_phi, camera_orig_theta) -> Tuple[Mobject, Mobject, Mobject]:
        curve, curve_extension = [
            ParametricFunction(
                lambda t: axes.coords_to_point(
                    t, np.exp(complex(0, PI*t)).real, np.exp(complex(0, PI*t)).imag),
                t_range=t_range,
            ) for t_range in [(0, 2, 0.1), (2, 4, 0.1)]]

        formula = MathTex(
            r"z = e^{i t \pi}, \quad t\in [0, 2]")
        formula.rotate(axis=OUT, angle=90 *
                       DEGREES).rotate(axis=UP, angle=90*DEGREES)
        formula.next_to(curve, UP + OUT)

        self.set_camera_orientation(
            phi=90*DEGREES, theta=0, focal_distance=10000)
        self.add(axes)
        self.play(Create(curve, run_time=2), Write(formula))
        self.wait()

        self.move_camera(phi=camera_orig_phi, theta=camera_orig_theta)
        self.wait()

        four = MathTex("4").rotate(
            axis=OUT, angle=90*DEGREES).rotate(axis=UP, angle=90*DEGREES)
        four.move_to(formula[0][12])
        self.play(Create(curve_extension, run_time=2),
                  formula[0][12].animate(run_time=1.5).become(four))

        return curve, curve_extension, formula

    def show_sin(self, curve, formula, camera_orig_phi, camera_orig_theta):
        self.move_camera(phi=90*DEGREES, theta=-90 *
                         DEGREES, focal_distance=10000)
        self.remove(formula)
        self.wait()
        sine_text = MathTex(
            r"\sin(t \pi) = \mathfrak{Im}(z)")
        sine_text.rotate(axis=RIGHT, angle=90 *
                         DEGREES).next_to(curve, 6*RIGHT + OUT)
        self.play(Write(sine_text), run_time=1)
        self.wait(4)

        # Overview
        self.add(formula)
        self.move_camera(phi=camera_orig_phi, theta=camera_orig_theta)
        self.play(FadeOut(sine_text, run_time=2))

    def show_cos(self, curve, formula, camera_orig_phi, camera_orig_theta):
        self.move_camera(phi=0, theta=-90*DEGREES, focal_distance=10000)
        self.remove(formula)
        self.wait()
        cosine_text = MathTex(r"\cos(t \pi) = \mathfrak{Re}(z)",
                              ).next_to(curve, 6*RIGHT + UP)
        self.play(Write(cosine_text), run_time=1)
        self.wait(4)

        # Overview
        self.add(formula)
        self.move_camera(phi=camera_orig_phi, theta=camera_orig_theta)
        self.play(FadeOut(cosine_text, run_time=2))

class Explanation(Scene):
    def construct(self):
        exp_sigma = MathTex(
            r"e^{x} = \sum_{k=0}^{\infty} \frac{x^k}{k!}")
        self.play(Write(exp_sigma))
        self.wait(2)

        exp_sum, exp_sum_substituted = self.substitute_exp_sum(
            exp_sigma=exp_sigma)
        self.wait()

        exp_sum_expanded = self.expand_exp_sum_substituted(
            exp_sum=exp_sum, exp_sum_substituted=exp_sum_substituted)
        self.wait()

        real_part, imag_part = self.extract_real_imag_parts(
            ref_sum=exp_sum_expanded,
            real_part_idxs=[1, slice(5, 7), slice(9, 11)],
            imag_part_idxs=[slice(3, 5), slice(7, 9)]
        )
        self.wait(3)

        self.play(FadeOut(exp_sum, exp_sum_expanded,
                  real_part, imag_part, shift=DOWN))
        self.wait()

        self.show_eulers_identity()
        self.wait(2)

    def substitute_exp_sum(self, exp_sigma: Mobject) -> Tuple[Mobject, Mobject]:
        exp_sum = [
            r"1 +", r"x", r"+ {", r"x", "^2 \over 2}", r" + {", r"x",
            r"^3 \over 6}", r" + {", r"x", r"^4 \over 24} + \dots"
        ]
        exp_sum_substituted = [el.replace("x", "(i t \pi)") for el in exp_sum]
        exp_sum_substituted[0] = r"z = " + exp_sum_substituted[0]

        exp_sum = MathTex(
            *exp_sum).next_to(exp_sigma, 2*DOWN)
        exp_sum_substituted = MathTex(*exp_sum_substituted)

        self.play(Write(exp_sum))
        self.wait(2)

        self.play(FadeOut(exp_sigma, shift=UP), exp_sum.animate.to_edge(UP))

        return exp_sum, exp_sum_substituted

    def expand_exp_sum_substituted(self, exp_sum: Mobject, exp_sum_substituted: Mobject) -> Mobject:
        exp_sum_copy = exp_sum.copy()
        self.play(exp_sum_copy.animate.shift(2*DOWN))
        self.wait()

        exp_sum_copy.set_color_by_tex("x", ORANGE)
        self.wait()

        exp_sum_substituted.set_color_by_tex("(i t \pi)", ORANGE)
        exp_sum_substituted.align_to(exp_sum_copy, UP)
        self.play(TransformMatchingTex(
            exp_sum_copy, exp_sum_substituted, run_time=2))
        self.wait()

        exp_sum_substituted_expanded = MathTex(*[
            r"z = 1 +", r"i", r"~t \pi + ", r"i^2", r"~{", r"(t \pi)^2", r" \over 2} + ",
            r"i^3", r"~{", r"(t \pi)^3", r" \over 6} + ", r"i^4", r"~{", r"(t \pi)^4",
            r" \over 24} + \dots"
        ]).align_to(exp_sum_substituted, UP)

        self.play(TransformMatchingShapes(
            exp_sum_substituted, exp_sum_substituted_expanded))
        self.wait()

        for i in [1, 3, 7, 11]:
            exp_sum_substituted_expanded[i].set_color(ORANGE)

        self.wait(2)

        exp_sum_substituted_expanded2 = MathTex(*[
            r"z = ", r"1", r" + ", r"i", r"~t \pi", r"-", r" { (t \pi)^2 \over 2}",
            r" - i", r"~{ (t \pi)^3 \over 6}", r" + ", r"{ (t \pi)^4 \over 24}", r" + \dots"
        ]).align_to(exp_sum_substituted, UP)

        exp_sum_color_idxs = [3, 5, 7, 9]

        for i in exp_sum_color_idxs:
            exp_sum_substituted_expanded2[i].set_color(ORANGE)

        self.play(TransformMatchingShapes(
            exp_sum_substituted_expanded, exp_sum_substituted_expanded2))
        self.wait(2)

        for i in exp_sum_color_idxs:
            exp_sum_substituted_expanded2[i]

        return exp_sum_substituted_expanded2

    def extract_real_imag_parts(self, ref_sum: Mobject,
                                real_part_idxs: List[Union[int, slice]],
                                imag_part_idxs: List[Union[int, slice]]
                                ) -> Tuple[Mobject, Mobject]:
        frameboxes_real = [
            SurroundingRectangle(ref_sum[i], buff=0.2, color=ORANGE)
            for i in real_part_idxs
        ]
        frameboxes_imag = [
            SurroundingRectangle(ref_sum[i], buff=0.2, color=ORANGE)
            for i in imag_part_idxs
        ]

        # Real Part
        real_part = VGroup(
            MathTex(r"\cos(t \pi) ="),
            MathTex(r"\mathfrak{Re}(z) = "),
            MathTex(
                r"1 - { (t \pi)^2 \over 2 } + { (t \pi)^4 \over 24} - \dots")
        ).arrange(RIGHT).next_to(ref_sum, 2.5*DOWN)

        self.play(LaggedStart(*[Create(box)
                  for box in frameboxes_real], lag_ratio=0.2))
        self.play(Write(real_part[1:]))
        self.wait()
        self.remove(*frameboxes_real)

        # Imag Part
        imag_part = VGroup(
            MathTex(r"\sin(t \pi) ="),
            MathTex(r"\mathfrak{Im}(z) = "),
            MathTex(r"t \pi - { (t \pi)^3 \over 6 } + \dots")
        ).arrange(RIGHT).align_to(real_part, LEFT + UP).shift(1.5*DOWN)

        self.play(LaggedStart(*[Create(box)
                  for box in frameboxes_imag], lag_ratio=0.2))
        self.play(Write(imag_part[1:]))
        self.wait()
        self.remove(*frameboxes_imag)

        self.play(Write(real_part[0]))
        self.play(Write(imag_part[0]))

        return real_part, imag_part

    def show_eulers_identity(self):
        eulers_identity = MathTex(
            r"e^{i t \pi} = \cos(t \pi) + i \sin(t \pi)")

        self.play(Write(eulers_identity))
        self.wait()

        eulers_identity_text = Text(
            "Euler's formula").next_to(eulers_identity, 3*UP)
        eulers_identity_box = SurroundingRectangle(
            eulers_identity, buff=0.3, color=ORANGE)
        self.play(Create(eulers_identity_box))
        self.wait()
        self.play(Write(eulers_identity_text))




