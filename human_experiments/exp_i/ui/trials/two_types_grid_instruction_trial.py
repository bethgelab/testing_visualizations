from psychopy import visual
from psychopy.visual import Window
from psychopy.event import Mouse
from .trial import Trial
from typing import List
from ui.grid_layout import GridLayout
from abc import ABC
import math
from typing import Optional, Tuple


class TwoTypesGridInstructionTrial(Trial, ABC):
    def __init__(
        self,
        mouse: Mouse,
        window: Window,
        a_stimuli: List[str],
        b_stimuli: List[str],
        a_title: str,
        b_title: str,
        stimulus_offset: Optional[Tuple[float, float]] = (0, 0)
    ):
        super().__init__(mouse, window)

        self.a_stimuli = a_stimuli
        self.b_stimuli = b_stimuli

        columns_a = math.ceil(math.sqrt(len(a_stimuli)))
        if columns_a > 0:
            rows_a = math.ceil(len(a_stimuli) / columns_a)
        else:
            rows_a = 0
        columns_b = math.ceil(math.sqrt(len(b_stimuli)))
        if columns_b > 0:
            rows_b = math.ceil(len(b_stimuli) / columns_b)
        else:
            rows_b = 0

        self.a_stimuli_grid = GridLayout((-stimulus_offset[0], +stimulus_offset[1]), -1, -1, columns_a, rows_a, 7)
        self.b_stimuli_grid = GridLayout((+stimulus_offset[0], +stimulus_offset[1]), -1, -1, columns_b, rows_b, 7)

        for item in a_stimuli:
            self.a_stimuli_grid.add_element(visual.ImageStim(self.window, item))
        for item in b_stimuli:
            self.b_stimuli_grid.add_element(visual.ImageStim(self.window, item))

        self.a_stimuli_grid.arrange()
        self.b_stimuli_grid.arrange()

        self.a_stimuli_grid.pos = (self.a_stimuli_grid.pos[0] - self.a_stimuli_grid.size[0] / 2, self.a_stimuli_grid.pos[1])
        self.b_stimuli_grid.pos = (self.b_stimuli_grid.pos[0] + self.b_stimuli_grid.size[0] / 2, self.b_stimuli_grid.pos[1])

        self.title_a_text = visual.TextStim(
            self.window, text=a_title, color="black", height=30
        )
        self.title_a_text.setPos((self.a_stimuli_grid.pos[0], self.a_stimuli_grid.size[1] / 2 + self.a_stimuli_grid.pos[1] + 20))

        self.title_b_text = visual.TextStim(
            self.window, text=b_title, color="black", height=30
        )
        self.title_b_text.setPos((self.b_stimuli_grid.pos[0], self.a_stimuli_grid.size[1] / 2 + self.b_stimuli_grid.pos[1] + 20))

    def draw(self):
        self.a_stimuli_grid.draw()
        self.b_stimuli_grid.draw()

        self.title_a_text.draw()
        self.title_b_text.draw()
