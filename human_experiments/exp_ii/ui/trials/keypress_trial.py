from .trial import Trial
from psychopy.visual import Window
from psychopy.event import Mouse
from psychopy.visual import TextStim
from psychopy import event
from typing import List


class KeypressTrial(Trial):
    def __init__(self, mouse: Mouse, window: Window, text: str, keys: List[str]):
        super().__init__(mouse, window)
        self.keys = keys
        self.instruction_text = TextStim(
            window, text=text, color="black", height=30, pos=(0, 0)
        )

    def run(self):
        self.draw()
        self.window.flip()

        keys = event.waitKeys(keyList=self.keys)
        self.window.flip()

        return keys

    def draw(self):
        self.instruction_text.draw(self.window)
