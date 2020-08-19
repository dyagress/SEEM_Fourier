"""Implement regular two dimensional Fourier grid."""

import numpy as np
import scipy
from interpolate import dirac
from matplotlib import pyplot as plt


class Grid:
    """Two dimensional Fourier grid with frequency data."""

    def __init(self, num_pts):
        self.num_pts = num_pts
        self.grid_length = 2*np.pi/num_pts
        # Physical grid.
        self.grid = 2*np.pi*np.mgrid[0:num_pts:1]/float(num_pts) - np.pi
        self.grid_x, self.grid_y = np.meshgrid(self.grid, self.grid,
                                               indexing='ij')
        # Frequency grid.
        self.freq = np.fft.fftfreq(num_pts, 1/float(num_pts))
        self.freq_x, self.freq_y = np.meshgrid(self.freq, self.freq,
                                               indexing='ij')

    def kernel(self, u, order=2):
        """Smoothing kernel of order p."""
        u = np.fft.fft2(u)
        u = u * (1 + self.grid_x**2 + self.grid_y**2)**(-order/2)
        u = np.fft.ifft2(u)
        u = np.real(u)
        return u

    def vector_kernel(self, u1, u2, order=2):
        """Smoothing kernel applied to vector field."""
        return self.kernel(u1, order), self.kernel(u2, order)
    
    def derivative(self, u,
                   order='laplacian',
                   accuracy='spectral'):
        """Discrete differentiation of u on regular grid."""
        if accuracy == 'spectral':
            u = np.fft.fft2(u)
            if order == 'laplacian':
                u *= -(self.grid_x**2 + self.grid_y**2)
            elif order == 'x':
                u *= 1j*self.grid_x
            elif order == 'y':
                u *= 1j*self.grid_y
            elif order == 'xx':
                u *= -self.grid_x**2
            elif order == 'xy':
                u *= -self.grid_x*self.grid_y
            elif order == 'yy':
                u *= -self.grid_y**2
            u = np.real(np.fft.ifft2(u))
            return u
        elif accuracy == 'cubic':
            if order == 'laplacian':
                u = (-4*u + np.roll(u, 1, axis=0) +
                     np.roll(u, -1, axis=0) + np.roll(u, 1, axis=1)
                     - np.roll(u, -1, axis=1)) / self.grid_length**2

    def interp(self, x, accuracy='spectral'):
        """Interpolation operator on regular grid."""
        if accuracy == 'spectral':
            return dirac(int(self.num_pts/2), self.grid, x.tolist()).flatten()
        elif accuracy == 'cubic':
            x0 = (x[0]+np.pi)/2/np.pi*self.num_pts
            x1 = (x[1]+np.pi)/2/np.pi*self.num_pts
            x0_range = np.arange(int(x0-1), int(x0+2))
            x1_range = np.arange(int(x1-1), int(x1+2))
            grid = (np.outer(x0_range, np.ones(4)).flatten() +
                    np.outer(np.ones(4), x1_range).flatten())
            x0_frac = x[0] % 1 + 1
            x1_frac = x[1] % 1 + 1
            coeff_x0 = np.array([-(x0_frac-1)*(x0_frac-2)*(x0_frac-3)/6,
                                 x0_frac*(x0_frac-2)*(x0_frac-3)/2,
                                 -x0_frac*(x0_frac-1)*(x0_frac-3)/2,
                                 x0_frac*(x0_frac-1)*(x0_frac-2)/6])
            coeff_x1 = np.array([-(x1_frac-1)*(x1_frac-2)*(x1_frac-3)/6,
                                 x1_frac*(x1_frac-2)*(x1_frac-3)/2,
                                 -x1_frac*(x1_frac-1)*(x1_frac-3)/2,
                                 x1_frac*(x1_frac-1)*(x1_frac-2)/6])
            coeff = np.outer(coeff_x0, coeff_x1).flatten()
            sparse = scipy.sparse.coo_matrix(coeff,
                                             (np.zeros(16), grid),
                                             shape=(1, self.num_pts**2))
            return sparse

    def div_free_projection(self, u1, u2):
        """Divergence free projection of vector field (u1, u2)."""
        f = self.freq_x**2 + self.freq_y**2
        fx = self.freq_x
        fy = self.freq_y
        f_u1 = np.fft.fft2(u1)
        f_u2 = np.fft.fft2(u2)
        a = fy*(fy*f_u1 - fx*f_u2)/f
        a[0, 0] = u1[0, 0]
        c = fx*(fx*f_u2 - fy*f_u1)/f
        c[0, 0] = u2[0, 0]
        v1 = np.real(np.fft.ifft2(a))
        v2 = np.real(np.fft.ifft2(c))
        return v1, v2

    def contour(self, w, n=50, figname='contour.pdf'):
        """Create contour plot for vector defined on regular grid."""
        plt.figure()
        plt.contourf(self.grid_x, self.grid_y, w, n)
        plt.axis('equal')
        plt.colorbar()
        plt.savefig(figname)
        plt.close('all')
