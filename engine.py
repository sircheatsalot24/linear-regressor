from collections.abc import Sequence
from typing import Self

import matplotlib.pyplot as plt
from numpy import average, linspace


class SplitLinearRegressor:
    points: Sequence[tuple[int | float, int | float]] | None
    _lefts: Sequence[tuple[int | float, int | float]] | None
    _rights: Sequence[tuple[int | float, int | float]] | None
    m: int | float | None = None
    b: int | float | None = None
    mae: float | int | None

    def __init__(self, m=None, b=None, points=None):
        self.m = m
        self.b = b
        self.points = points
        self._lefts, self._rights = None, None
        self.mae = None

    def train(self, points: Sequence[tuple[int | float, int | float]]) -> Self:

        self.points = points
        xs = [pt[0] for pt in points]

        lefts = []
        rights = []
        for pt in points:
            if pt[0] <= ((max(xs) + min(xs)) / 2):
                lefts.append(pt)
            else:
                rights.append(pt)

        self._lefts, self._rights = lefts, rights

        leftsx = [p[0] for p in lefts]
        rightsx = [p[0] for p in rights]

        leftsy = [p[1] for p in lefts]
        rightsy = [p[1] for p in rights]

        avgleft = (average(leftsx), average(leftsy))
        avgright = (average(rightsx), average(rightsy))

        x1, y1, x2, y2 = avgleft[0], avgleft[1], avgright[0], avgright[1]

        n1 = y2 - y1
        n2 = x2 - x1
        m = n1 / n2
        b = -m * x1 + y1

        def f(x):
            return m * x + b

        mae = 0
        for pt in points:
            mae += abs(f(pt[0]) - pt[1])
        mae /= len(xs)

        self.m = m
        self.b = b
        self.mae = mae

        return self

    def predict(self, x: list | float) -> float | list:
        if self.m is None or self.b is None:
            if self.points is not None:
                self.train(self.points)
                assert self.m is not None
                assert self.b is not None
            else:
                raise AttributeError(
                    "Slope and y-intercept not found! Use .train() first, "
                    "or construct with points."
                )

        if isinstance(x, list):
            return [self.m * xs + self.b for xs in x]
        else:
            return self.m * x + self.b

    def show_plot(self):
        if self._lefts is None or self._rights is None or self.points is None:
            raise AttributeError("Use .train() to construct a line before plotting!")
        for left in self._lefts:
            plt.plot(left[0], left[1], "ro")
        for right in self._rights:
            plt.plot(right[0], right[1], "bo")
        leftsx = [p[0] for p in self._lefts]
        rightsx = [p[0] for p in self._rights]

        leftsy = [p[1] for p in self._lefts]
        rightsy = [p[1] for p in self._rights]

        avgleft = (average(leftsx), average(leftsy))
        avgright = (average(rightsx), average(rightsy))

        x1, y1, x2, y2 = avgleft[0], avgleft[1], avgright[0], avgright[1]

        xs = [pt[0] for pt in self.points]
        xline = linspace(min(xs), max(xs), 500)
        yline = [self.predict(x) for x in xline]

        plt.scatter(xline, yline)

        plt.plot(x1, y1, "go")
        plt.plot(x2, y2, "go")

        plt.show()

    def __repr__(self):
        if self.m is not None and self.b is not None and self.mae is not None:
            return f"SplitLinearRegressor: y = {self.m:.2f}x {'+' if self.b >= 0 else '-'} {abs(self.b):.2f} / Mean Absolute Error (MAE): {self.mae:.2f}"
        else:
            return "SplitLinearRegressor: no line defined. Use .train() with points to fit a line."
