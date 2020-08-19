"""Implement boundary and interior discretization on regular Fourier grid."""

import numpy as np
from grid import Grid


class Domain:
    """Domain discretization (interior and boundary) on regular grid."""

    def __init__(self, grid:Grid,
                 interior_test,
                 boundary,
                 density=1):
        self.grid = grid
        # Interior.
        self.interior_flag = interior_test(grid.grid_x, grid.grid_y)
        self.num_int_pts = np.sum(interior_flag)
        # Boundary.
