from psychopy import visual, core
from psychopy.visual import Window
from psychopy.event import Mouse
from ui.trials import TwoTypesGridInstructionTrial
from typing import List


class WarmUpTrial(TwoTypesGridInstructionTrial):
    def __init__(
        self,
        mouse: Mouse,
        window: Window,
        a_stimuli: List[str],
        b_stimuli: List[str],
        a_title: str,
        b_title: str,
    ):
        super().__init__(mouse, window, a_stimuli, b_stimuli, a_title, b_title, (480, 150))

        self.slider: visual.Slider = visual.Slider(
            self.window,
            pos=(0, -400),#-250),
            ticks=(-100, 100),
            labels=("Not intuitive", "Intuitive"),
            granularity=1,
            color="black",
            units="pix",
            size=(600, 20),
            style="rating",
        )

        self.slider.markerPos = 0
        self.slider.marker.size = (20, 20)
        self.slider.marker.color = "black"
        self.slider.mouse = self.mouse

        self.instructions_text = visual.TextStim(
            self.window,
            text="How intuitive do you find\n"
            "the explanation of\n"
            "these optimized images\n"
            "for the given natural images?",
            color="black",
            height=0.8 * 30,
        )
        self.instructions_text.setPos((0, -300))#-150))

    def draw(self):
        super().draw()

        self.slider.draw()
        self.instructions_text.draw()

    def run(self):
        """ Method for one warm-up trial"""
        self.slider.reset()
        self.slider.markerPos = 0
        self.mouse.setPos(self.slider.pos)
        self.mouse.setVisible(True)

        while not self.slider.rating:
            self.draw()
            self.window.flip()

        core.wait(1)
        # hide stimuli
        self.a_stimuli_grid.opacity = 0.0
        self.b_stimuli_grid.opacity = 0.0
        self.draw()
        self.window.flip()
        core.wait(0.1)

        return self.slider.rating, self.slider.rt
