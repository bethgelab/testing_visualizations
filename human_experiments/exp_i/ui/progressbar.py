from psychopy import visual


class ProgressBar:
    """This class represents a progress bar which shows the fraction of completed trials during the experiment"""

    def __init__(self, window, pos, width=20, height=20):
        self.window = window
        self.pos = pos
        self.width = width

        # create outline of the progress bar
        self.outer = visual.Rect(
            self.window,
            pos=self.pos,
            width=width,
            height=height,
            lineWidth=3,
            lineColor="black",
        )

        # create inner part which is supposed to move as time progresses
        self.inner = visual.Rect(
            self.window,
            pos=(self.pos[0], self.pos[1]),
            width=0,
            height=height - 2,
            opacity=0.25,
            lineWidth=0,
            fillColor="white",
        )

    def reset(self):
        self.update(0)

    @property
    def opacity(self):
        return self.outer.opacity

    @opacity.setter
    def opacity(self, value):
        self.outer.opacity = value
        self.inner.opacity = value

    def update(self, current):
        # current: in range [0, 1]

        current_width = current * self.width
        self.inner.pos[0] = self.pos[0] + (-(self.width / 2) + (current_width / 2))
        self.inner.width = current_width

    def draw(self):
        self.outer.draw(self.window)
        self.inner.draw(self.window)
