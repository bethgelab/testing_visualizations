from ui.layout import Layout
import math
from typing import Tuple


class GridLayout(Layout):
    def __init__(
        self,
        position: Tuple[float, float],
        item_width: float,
        item_height: float,
        n_rows: int = -1,
        n_columns: int = -1,
        margin: float = 0,
    ):
        super().__init__(position)

        assert n_rows != -1 or n_columns != -1, "At least one dimension must be set"

        self.item_width = item_width
        self.item_height = item_height
        self.margin = margin

        self.n_rows = n_rows
        self.n_columns = n_columns

    def add_element(self, element):
        self._elements.append(element)

    @property
    def _effective_item_width(self):
        if self.item_width != -1:
            item_width = self.item_width
        else:
            if len(self._elements) == 0:
                item_width = 1
            else:
                item_width = max([it.size[0] for it in self._elements])

        return item_width

    @property
    def _effective_item_height(self):
        if self.item_height != -1:
            item_height = self.item_height
        else:
            if len(self._elements) == 0:
                item_height = 1
            else:
                item_height = max([it.size[1] for it in self._elements])

        return item_height

    def arrange(self):
        n_rows = self.n_rows
        n_columns = self.n_columns

        n_elements = len(self._elements)

        item_width = self._effective_item_width
        item_height = self._effective_item_height
        
        if n_rows == -1:
            n_rows = int(math.ceil(n_elements / n_columns))

        if n_columns == -1:
            n_columns = int(math.ceil(n_elements / n_rows))

        width = n_columns * item_width + (n_columns - 1) * self.margin
        height = n_rows * item_height + (n_rows - 1) * self.margin
        offset_x = -width / 2 + item_width / 2
        offset_y = -height / 2 + item_width / 2

        positions = []
        for i in range(len(self._elements)):
            row = i // n_columns
            column = i % n_columns

            positions.append(
                (
                    self.pos[0] + offset_x + row * (item_width + self.margin),
                    self.pos[1] + offset_y + column * (item_height + self.margin),
                )
            )

        for position, element in zip(positions, self._elements):
            element.pos = position