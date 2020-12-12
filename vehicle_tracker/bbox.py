#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Milan Ondrasovic <milan.ondrasovic@gmail.com>

from typing import Iterator, Tuple


class BBox:
    """
    Bounding box with integer coordinates in a 2D image.
    """
    
    def __init__(self, x: int, y: int, width: int, height: int) -> None:
        """
        :param x: x coordinate
        :param y: y coordinate
        :param width: a positive width
        :param height: a positive height
        """
        assert (width > 0) and (height > 0)
        
        self.x, self.y, self.width, self.height = x, y, width, height
        
        self._center = (self.x + int(round(self.width / 2)),
                        self.y + int(round(self.height / 2)))
    
    def __iter__(self) -> Iterator[int]:
        return iter((self.x, self.y, self.width, self.height))
    
    def __eq__(self, other) -> bool:
        return isinstance(other, BBox) and \
               ((self.x == other.x) and (self.y == other.y) and
                (self.width == other.width) and (self.height == other.height))
    
    def __hash__(self) -> int:
        return hash((self.x, self.y, self.width, self.height))
    
    def __repr__(self) -> str:
        return f'BBox({self.x},{self.y},{self.width},{self.height})'
    
    @property
    def top_left(self) -> Tuple[int, int]:
        return self.x, self.y
    
    @property
    def top_right(self) -> Tuple[int, int]:
        return self.x + self.width, self.y
    
    @property
    def bottom_left(self) -> Tuple[int, int]:
        return self.x, self.y + self.height
    
    @property
    def bottom_right(self) -> Tuple[int, int]:
        return self.x + self.width, self.y + self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return self.x + self.width // 2, self.y + self.height // 2
    
    def area(self) -> int:
        return self.width * self.height
    
    def intersection_bbox(self, other: 'BBox') -> 'BBox':
        top_left_x = max(self.x, other.x)
        top_left_y = max(self.y, other.y)
        bottom_right_x = min(self.x + self.width, other.x + other.width)
        bottom_right_y = min(self.y + self.height, other.y + other.height)
        
        width = bottom_right_x - top_left_x
        height = bottom_right_y - top_left_y
        
        if min(width, height) <= 0:
            raise ValueError('bounding boxes have no intersection')
        
        return BBox(top_left_x, top_left_y, width, height)
    
    def intersection_area(self, other: 'BBox') -> int:
        a = (min(self.x + self.width, other.x + other.width) -
             max(self.x, other.x))
        b = (min(self.y + self.height, other.y + other.height) -
             max(self.y, other.y))
        
        return max(0, a) * max(0, b)
    
    def intersection_over_union(self, other: 'BBox') -> float:
        intersection_area = self.intersection_area(other)
        union_area = self.area() + other.area() - intersection_area
        return intersection_area / float(union_area)

    def rescale_sides(
            self, width_scale: float = 1., height_scale: float = 1.) -> 'BBox':
        assert (width_scale >= 0) and (height_scale >= 0)
    
        x_diff = int(self.width * (width_scale - 1.) / 2)
        y_diff = int(self.height * (height_scale - 1.) / 2)
    
        new_x, new_y = self.x - x_diff, self.y - y_diff
        new_w, new_h = self.width + x_diff, self.height + y_diff
    
        return BBox(new_x, new_y, new_w, new_h)
