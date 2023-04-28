import numpy as np
import random
from itertools import chain

class RandomGridGenerator:
    def __init__(self, rown, coln=0):
        """Set the length according to the size of idxs and cols"""
        if rown < 3:
            raise ValueError("idx length should be at least 3")

        self.rown = rown
        self.coln = coln

    def getEmptyGrid(self):
        """Generates an empty row with initial idx size. All 0s by default"""
        base = lambda: [0] * self.rown
        return np.array([base() for i in range(self.coln)], dtype= np.int64)

    def _fillRandomRow(self, row: list, elements: list, density_threshold, count=0):
        if count > self.rown:
            raise ValueError("Count should be less than the line length")

        spots = set()

        while len(spots) < count:
            spots.add(random.randint(0, self.rown - 1))

        for i in spots:
            if random.random() > density_threshold:
                row[i] = random.choice(elements)

    def getGrid(
        self, elements: list[int], density_threshold: float = 0.8, max_per_row=None
    ):
        """Randomly generates a grid with some possibility of items from
        elements ocurring."""
        base = self.getEmptyGrid()
        max_per_row = max_per_row or self.coln

        for idx, row in enumerate(base):
            self._fillRandomRow(row, elements, density_threshold, max_per_row)

        return base

class Grid:

    def __init__(self, starting_grid):
        self.grid = np.array(starting_grid, dtype= np.int64)
        self.rown = len(starting_grid[0])
        self.coln = len(starting_grid)

    def _get_adjacent_with_dist(self, row, col, dist):
        """assumes that the distance is small enough compared with edges to avoid overlapping indexes."""
        c_lambda = lambda c: c % self.coln
        r_lambda = lambda r: r % self.rown

        right = map(c_lambda, range(col+1, col+dist))
        left = map(c_lambda, range(col-dist+1, col))

        top = map(r_lambda, range(row-dist+1, row))
        bottom = map(r_lambda, range(row+1, row+dist))

        return chain(top, [row], bottom), chain(left, [col], right)

    def getNeighbors(self, row, col, dist):
        ri, ci = self._get_adjacent_with_dist(row, col, dist)
        return self.grid[list(ri), :][:, list(ci)]

    def getNeighborsFromNext(self, _next, row, col, dist):
        ri, ci = self._get_adjacent_with_dist(row, col, dist)
        return _next[list(ri), :][:, list(ci)]
