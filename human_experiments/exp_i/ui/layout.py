from abc import ABC, abstractmethod
from psychopy import visual
from typing import List, Tuple
import numpy as np


class Layout(ABC):
    def __init__(self, position: Tuple[float, float]):
        self._position: Tuple[float, float] = position
        self._elements: List[visual.BaseVisualStim] = []

    @property
    def opacity(self):
        return NotImplementedError()

    @opacity.setter
    def opacity(self, value):
        for element in self._elements:
            element.opacity = value

    @abstractmethod
    def arrange(self):
        pass

    def draw(self):
        for element in self._elements:
            element.draw()

    @property
    def pos(self):
        return self._position

    @pos.setter
    def pos(self, value):
        self._position = value
        self.arrange()

    @property
    def size(self):
        if len(self._elements) == 0:
            return np.array([0, 0])
            
        min_x = min([it.pos[0] - it.size[0] / 2 for it in self._elements])
        max_x = max([it.pos[0] + it.size[0] / 2 for it in self._elements])

        min_y = min([it.pos[1] - it.size[1] / 2 for it in self._elements])
        max_y = max([it.pos[1] + it.size[1] / 2 for it in self._elements])

        width = max_x - min_x
        height = max_y - min_y

        return np.array([width, height])
