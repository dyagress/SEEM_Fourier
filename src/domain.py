"""Implement boundary and interior discretization on regular Fourier grid."""

import numpy as np
from src.grid import Grid
from scipy.sparse import vstack as sparse_vstack

def space_points(bdry_x, bdry_y, dx):
    """Space points equally on boundary with distance approximately dx."""
    m = 8192
    k = np.arange(m+1)/m
    fine_boundary = np.transpose([bdry_x(k), bdry_y(k)])
    # Find length of boundary curve.
    pts_diff = fine_boundary[1:] - fine_boundary[:-1]
    ptwise_len = np.linalg.norm(pts_diff, axis=1)
    length = np.sum(ptwise_len)
    num_pts = int(length/dx)
    corrected_dx = length/num_pts
    # Place points corrected_dx apart on boundary.
    boundary = np.array([bdry_x(0), bdry_y(0)])
    current_position = 0
    current_index = 1
    current_fine_index = 1
    while current_index < num_pts:
        current_position += ptwise_len[current_fine_index]
        current_fine_index += 1
        if current_position >= current_index*corrected_dx:
            boundary = np.vstack((boundary,
                                  fine_boundary[current_fine_index, :]))
            current_index += 1
    return boundary

def vstack(vectors):
    if type(vectors[0]) == scipy.sparse.coo.coo_matrix:
        return sparse_vstack(vectors)
    else:
        return np.vstack(vectors)

class Domain:
    """Domain discretization (interior and boundary) on regular grid."""

    def __init__(self, grid: Grid,
                 interior_test,
                 boundary,
                 boundary_density=1,
                 interp_accuracy='cubic'):
        """Create discretization of domain, interior and boundary.

        Every domain is associated with a regular grid, grid.
        interior_test takes x and y values as input and determines if a
        point is in the interior of the domain.
        Boundary is a list of curve parametrizations which parametrize the
        full boundary.
        boundary_density measures the ratio of the density of the boundary
        discretization to the density of the regular grid.
        It can be decreased less than one to improve conditioning.
        """    
        self.grid = grid
        # Interior.
        self.interior_flag = interior_test(grid.grid_x, grid.grid_y)
        self.num_int_pts = np.sum(self.interior_flag)
        # Boundary.
        self.boundary_density = boundary_density
        self.dx = self.grid.grid_length / boundary_density
        self.boundary = space_points(boundary[0],
                                     boundary[1],
                                     self.dx)
        self.num_bdry_pts = self.boundary.shape[0]
        self.interp_accuracy = interp_accuracy
        boundary_interpolation = []
        for i in range(self.num_bdry_pts):
            boundary_interpolation.append(self.grid.interp(self.boundary[i, :],
                                                           interp_accuracy))
        self.interp = vstack(boundary_interpolation)

        def restriction(self, u):
            return u[self.interior_flag]

        def extension_by_zero(self, u):
            u_ext = np.zeros_like(self.grid.grid_x)
            u_ext[self.interior_flag] = u
            return u_ext

        
