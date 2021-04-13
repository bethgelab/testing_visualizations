from ui.trials import TwoTypesGridInstructionTrial
from psychopy.event import Mouse
from psychopy.visual import Window
from typing import List, Optional
from typing_extensions import Literal
from ui import HorizontalUncertainTwoAFCStim, VerticalUncertainTwoAFCStim
from psychopy.visual import ImageStim
from psychopy import core
from ui import ProgressBar
import numpy as np


class ForcedChoiceTrial(TwoTypesGridInstructionTrial):
    """ Method for one trial of a 2-AFC task with confidence judgment """

    def __init__(
        self,
        mouse: Mouse,
        window: Window,
        a_stimuli: List[str],
        b_stimuli: List[str],
        a_title: str,
        b_title: str,
        a_stimulus: str,
        b_stimulus: str,
        orientation: Optional[Literal["vertical", "horizontal"]] = "vertical",
        timeout: Optional[float] = None,
        instructions: Optional[str] = None,
        correct: Optional[Literal["a", "b"]] = None,
        global_progress: Optional[float] = None,
    ):
        assert orientation in ("vertical", "horizontal")
        if orientation == "horizontal":
            grid_offset = (57, 0)
        else:
            grid_offset = (240, 0)

        super().__init__(mouse, window, a_stimuli, b_stimuli, a_title, b_title, grid_offset)

        self.instructions = instructions
        self.timeout = timeout

        a_stimulus_image = ImageStim(window, a_stimulus)
        b_stimulus_image = ImageStim(window, b_stimulus)

        self.orientation = orientation
        if orientation == "horizontal":
            self.uncertain_2afc = HorizontalUncertainTwoAFCStim(
                mouse,
                window,
                a_stimulus_image,
                b_stimulus_image,
                pos=(0, -350),
                correct=correct,
            )
        else:
            self.uncertain_2afc = VerticalUncertainTwoAFCStim(
                mouse,
                window,
                a_stimulus_image,
                b_stimulus_image,
                pos=(0, 0),
                correct=correct,
            )

        self.timeout = timeout
        if timeout is None:
            self.timeout_progress_bar = None
        else:
            self.timeout_progress_bar = ProgressBar(window, pos=(0, -532), height=14, width=2 * (90 + a_stimulus_image.size[0]))

        if global_progress is None:
            self.global_progress_bar = None
        else:
            self.global_timeout_progress_bar = ProgressBar(window, pos=(0, -500), height=20, width=window.size[0])
            self.global_timeout_progress_bar.update(global_progress)

    def run(self):
        """ Method for one 2AFC trial"""
        self.uncertain_2afc.reset()
        if self.orientation == "horizontal":
            self.mouse.setPos(self.uncertain_2afc.pos + np.array([0, -115]))
        else:
            self.mouse.setPos(self.uncertain_2afc.pos + np.array([0, 0]))

        self.mouse.setVisible(True)

        start_time = core.getTime()

        # wait until the participant clicked on one of the patches or the
        # timeout is reached
        timeout_reached = False
        while not self.uncertain_2afc.response:
            self.draw()
            self.window.flip()

            time_passed = core.getTime() - start_time
            if self.timeout_progress_bar is not None and not timeout_reached:
                self.timeout_progress_bar.update(time_passed / self.timeout)

                if time_passed >= self.timeout:
                    timeout_reached = True

                    self.timeout_progress_bar.inner.fillColor = "red"

        rt = self.uncertain_2afc.rt - start_time
        response = self.uncertain_2afc.response
        rating = self.uncertain_2afc.rating

        # wait for just a tiny moment, so that the feedback does not appear too fast
        core.wait(0.50)
        self.uncertain_2afc.highlight_correct()
        if self.timeout_progress_bar is not None:
            self.timeout_progress_bar.opacity = 0.0
        self.draw()
        self.window.flip()
        # show feedback for 1.5 seconds
        core.wait(1.5)

        # hide stimuli
        self.a_stimuli_grid.opacity = 0.0
        self.b_stimuli_grid.opacity = 0.0
        self.uncertain_2afc.reset()
        if self.timeout_progress_bar is not None:
            self.timeout_progress_bar.reset()
        self.draw()
        self.window.flip()
        core.wait(0.1)

        return rt, response, rating, timeout_reached

    def draw(self):
        super().draw()
        self.uncertain_2afc.draw()
        if self.timeout_progress_bar is not None:
            self.timeout_progress_bar.draw()
        if self.global_timeout_progress_bar is not None:
            self.global_timeout_progress_bar.draw()
