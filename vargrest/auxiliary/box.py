from typing import Tuple


class Box:
    def __init__(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    def loc(self) -> Tuple[float, float]:
        return 0.5 * (self.x0 + self.x1), 0.5 * (self.y0 + self.y1)

    def side_lengths(self) -> Tuple[float, float]:
        return abs(self.x1 - self.x0), abs(self.y1 - self.y0)

    @staticmethod
    def create_using_center(center_x: float, center_y: float, length_x: float, length_y: float) -> 'Box':
        return Box(center_x - 0.5 * length_x,
                   center_y - 0.5 * length_y,
                   center_x + 0.5 * length_x,
                   center_y + 0.5 * length_y)

    def __repr__(self) -> str:
        return f'(x0={self.x0},y0={self.y0},x1={self.x1},y1={self.y1})'
