from abc import ABC, abstractmethod
from psychopy.visual import Window
from psychopy.event import Mouse


class Trial(ABC):
    def __init__(self, mouse: Mouse, window: Window):
        self.mouse = mouse
        self.window = window

    @abstractmethod
    def run(self):
        pass
