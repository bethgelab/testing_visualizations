from psychopy.event import Mouse
from psychopy.visual import Window
from typing import List
from .trial import Trial


class BatchedTrial(Trial):
    def __init__(self, mouse: Mouse, window: Window, trials: List[Trial]):
        super().__init__(mouse, window)

        self.trials = trials

    def run(self):
        for trial in self.trials:
            trial.run()

    def draw(self):
        pass
