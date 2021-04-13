from psychopy.visual import BaseVisualStim, TextStim
from psychopy.visual.shape import ShapeStim
from psychopy.visual.rect import Rect
from typing import List, Optional
from typing_extensions import Literal
from abc import ABC, abstractmethod


class UncertainTwoAFCStim(BaseVisualStim, ABC):
    def __init__(
        self,
        mouse,
        win,
        stimulus_a: BaseVisualStim,
        stimulus_b: BaseVisualStim,
        pos=None,
        correct: Optional[Literal["a", "b"]] = None,
        granularity: int = 3,
    ):
        super().__init__(win=win)
        self.mouse = mouse

        self.correct = correct
        self.granularity = granularity

        self.__dict__["pos"] = (0, 0)

        self._stimulus: List[BaseVisualStim] = []
        self._elements: List[BaseVisualStim] = []

        self._initialize_elements(win, stimulus_a, stimulus_b)

        if pos is not None:
            self.pos = pos

        self.response = None
        self.rt = None
        self.rating = None

    @abstractmethod
    def _initialize_elements(self, win, stimulus_a, stimulus_b):
        pass

    @property
    def pos(self, **kwargs):
        return self.__dict__["pos"]

    @pos.setter
    def pos(self, value):
        if value is None:
            value = (0, 0)

        diff_x = value[0] - self.pos[0]
        diff_y = value[1] - self.pos[1]
        for element in self._elements:
            element.pos = (element.pos[0] + diff_x, element.pos[1] + diff_y)

        self.__dict__["pos"] = value

    def draw(self):
        self._update_response()

        # draw stimuli
        self.stimulus_a.draw()
        self.stimulus_b.draw()

        # draw rectangles
        self.choice_text_rectangle.draw()
        self.choice_stimulus_rectangle.draw()
        self.correct_rectangle.draw()

        # draw certainty bar
        self.arrow_a_text.draw()
        self.arrow_a_shape.draw()
        self.arrow_b_text.draw()
        self.arrow_b_shape.draw()

        for element in self.arrow_levels_text:
            element.draw()

    def _update_response(self):
        # don't change anything once the user gave an answer
        if self.response is not None:
            return

        buttons, rt = self.mouse.getPressed(getTime=True)
        rt = rt[0]

        # check if the left mouse button got pressed
        if buttons[0] == 1:
            # check if the participant clicked on one of the patches
            for rating in self.arrow_levels_text:
                if self.mouse.isPressedIn(rating.clickable_area):
                    # confidence = int(rating.text)
                    confidence = rating.value
                    self.choice_text_rectangle.opacity = 1.0
                    self.choice_text_rectangle.setPos(rating.pos)

                    if int(confidence) < 0:  # if the conf rating is -3,-2 or -1
                        # choose left stimulus
                        stimulus_clicked = self.stimulus_a
                        response = "a"
                    else:  # if conf rating is 1,2 or 3
                        # choose right stimulus
                        stimulus_clicked = self.stimulus_b
                        response = "b"

                    self.choice_stimulus_rectangle.opacity = 1.0
                    self.choice_stimulus_rectangle.setPos(stimulus_clicked.pos)

                    self.rt = rt
                    self.response = response
                    self.rating = confidence

                    return
                else:
                    continue

    def highlight_correct(self):
        # give feedback whether the clicked patch was the correct one
        if self.correct is not None:
            self.correct_rectangle.opacity = 1.0
            if self.correct == "a":
                # if correct: put a green frame
                self.correct_rectangle.setPos(self.stimulus_a.pos)
            elif self.correct == "b":
                self.correct_rectangle.setPos(self.stimulus_b.pos)

    def reset(self):
        self.choice_stimulus_rectangle.opacity = 0.0
        self.choice_text_rectangle.opacity = 0.0
        self.correct_rectangle.opacity = 0.0

        self.response = None
        self.rt = None
        self.rating = None


class HorizontalUncertainTwoAFCStim(UncertainTwoAFCStim):
    def _initialize_elements(self, win, stimulus_a, stimulus_b):
        stimulus_a.pos = (-stimulus_a.size[0] / 2 - 90, +5)
        stimulus_b.pos = (+stimulus_b.size[0] / 2 + 90, +5)

        self.stimulus_a = stimulus_a
        self.stimulus_b = stimulus_b

        self._elements.append(stimulus_a)
        self._elements.append(stimulus_b)

        # create arrows for confidence rating
        # create vertices for the arrow
        arrow_shape_vertices = [
            (0, 0),
            (230, 0),
            (220, 5),
            (230, 0),
            (220, -5),
        ]
        self.arrow_a_shape = ShapeStim(
            win,
            vertices=arrow_shape_vertices,
            size=1,
            lineColor="black",
            lineWidth=3,
            closeShape=False,
            pos=(+90, -150),
        )
        self.arrow_b_shape = ShapeStim(
            win,
            vertices=arrow_shape_vertices,
            size=1,
            lineColor="black",
            lineWidth=3,
            closeShape=False,
            ori=180,
            pos=(-90, -150),
        )
        self._elements.append(self.arrow_a_shape)
        self._elements.append(self.arrow_b_shape)

        # hide the rectangles until they are needed (opacity=0)
        self.choice_text_rectangle = Rect(
            win, 40, 40, lineWidth=5, lineColor="black", opacity=0.0
        )
        self.choice_stimulus_rectangle = Rect(
            win,
            stimulus_a.size[1],
            stimulus_a.size[0],
            lineWidth=5,
            lineColor="black",
            opacity=0.0,
        )
        self.correct_rectangle = Rect(
            win,
            stimulus_a.size[1] + 7.5,
            stimulus_a.size[0] + 7.5,
            lineWidth=5,
            lineColor="green",
            opacity=0.0,
        )
        self._elements.append(self.choice_text_rectangle)
        self._elements.append(self.choice_stimulus_rectangle)
        self._elements.append(self.correct_rectangle)

        self.arrow_a_text = TextStim(
            win,
            text="More confident",
            color="black",
            height=18,
            anchorHoriz="left",
            alignText="left",
            pos=(-270, -160),
        )
        self.arrow_b_text = TextStim(
            win,
            text="More confident",
            color="black",
            height=18,
            anchorHoriz="right",
            alignText="right",
            pos=(+270, -160),
        )
        self._elements.append(self.arrow_a_text)
        self._elements.append(self.arrow_b_text)

        self.arrow_levels_text: List[TextStim] = []
        arrow_levels_values = [-3, -2, -1, 1, 2, 3]
        arrow_levels_texts = [str(x) if x > 0 else str(-x) for x in arrow_levels_values]
        arrow_levels_x_positions = [-300, -200, -100, +100, +200, +300]
        for position_x, text, value in zip(arrow_levels_x_positions, arrow_levels_texts, arrow_levels_values):
            item = TextStim(
                win, text=text, color="black", height=25, pos=(position_x, -130)
            )
            rect = Rect(win, width=25, height=25, lineWidth=0.0, pos=(position_x, -130))
            item.value = value
            item.clickable_area = rect
            self.arrow_levels_text.append(item)
            self._elements.append(item)
            self._elements.append(rect)


class VerticalUncertainTwoAFCStim(UncertainTwoAFCStim):
    def _initialize_elements(self, win, stimulus_a, stimulus_b):
        stimulus_a.pos = (0, -stimulus_a.size[0] / 2 - 90)
        stimulus_b.pos = (0, +stimulus_b.size[0] / 2 + 90)

        self.stimulus_a = stimulus_a
        self.stimulus_b = stimulus_b

        self._elements.append(stimulus_a)
        self._elements.append(stimulus_b)

        # create arrows for confidence rating
        # create vertices for the arrow
        arrow_shape_vertices = [
            (0, 0),
            (230, 0),
            (220, 8),
            (230, 0),
            (220, -8),
        ]
        self.arrow_a_shape = ShapeStim(
            win,
            vertices=arrow_shape_vertices,
            size=1,
            lineColor="black",
            lineWidth=3,
            closeShape=False,
            pos=(-112, 25),
        )
        self.arrow_b_shape = ShapeStim(
            win,
            vertices=arrow_shape_vertices,
            size=1,
            lineColor="black",
            lineWidth=3,
            closeShape=False,
            pos=(-112, -25),
        )
        self._elements.append(self.arrow_a_shape)

        # hide the rectangles until they are needed (opacity=0)
        self.choice_text_rectangle = Rect(
            win, 40, 40, lineWidth=5, lineColor="black", opacity=0.0
        )
        self.choice_stimulus_rectangle = Rect(
            win,
            stimulus_a.size[1],
            stimulus_a.size[0],
            lineWidth=5,
            lineColor="black",
            opacity=0.0,
        )
        self.correct_rectangle = Rect(
            win,
            stimulus_a.size[1] + 7.5,
            stimulus_a.size[0] + 7.5,
            lineWidth=5,
            lineColor="green",
            opacity=0.0,
        )
        self._elements.append(self.choice_text_rectangle)
        self._elements.append(self.choice_stimulus_rectangle)
        self._elements.append(self.correct_rectangle)

        self.arrow_a_text = TextStim(
            win,
            text="More confident",
            color="black",
            height=25,
            anchorHoriz="center",
            alignText="center",
            pos=(0, +0),
        )
        self.arrow_b_text = self.arrow_a_text
        self._elements.append(self.arrow_a_text)

        self.arrow_levels_text: List[TextStim] = []
        arrow_levels_values = [1, 2, 3, -1, -2, -3]
        arrow_levels_texts = [str(x) if x > 0 else str(-x) for x in arrow_levels_values]
        arrow_levels_positions = [(-100, 60), (0, 60), (+100, 60), (-100, -60), (0, -60), (+100, -60)]
        for (position_x, position_y), text, value in zip(arrow_levels_positions, arrow_levels_texts, arrow_levels_values):
            item = TextStim(
                win, text=text, color="black", height=25, pos=(position_x, position_y)
            )
            rect = Rect(win, width=25, height=25, lineWidth=0.0, pos=(position_x, position_y))
            item.value = value
            item.clickable_area = rect
            self.arrow_levels_text.append(item)
            self._elements.append(item)
            self._elements.append(rect)
