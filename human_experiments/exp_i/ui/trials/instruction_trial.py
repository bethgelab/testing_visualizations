from .trial import Trial
from psychopy.visual import Window
from psychopy.event import Mouse
from psychopy.visual import TextStim
from psychopy import core


class InstructionTrial(Trial):
    """pseudo trial to display text"""

    def __init__(self, mouse: Mouse, window: Window, text: str, duration: float):
        super().__init__(mouse, window)
        self.duration = duration
        self.instruction_text = TextStim(
            window, text=text, color="black", height=30, pos=(0, 0)
        )

    def run(self):
        self.draw()
        self.window.flip()
        core.wait(self.duration)
        self.window.flip()

    def draw(self):
        self.instruction_text.draw(self.window)
