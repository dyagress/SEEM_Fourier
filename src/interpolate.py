import numpy as np

def dirac(m, x, x0):
    """Dirac computed at resolution m but restricted 
    to a subinterval determined by x"""
    if type(x0) != list:
        a = np.sinc((float(m) - .5) * (x - x0) / np.pi)
        b = np.sinc(.5 * (x - x0) / np.pi)
        d = (m - .5) * a / b + .5 * np.cos(m * (x - x0))
        return d / float(m)
    elif type(x0) == list and len(x0) == 2:
        return np.outer(dirac(m, x, x0[0]), dirac(m, x, x0[1]))
    else:
        print('Invalid input for Dirac')
        return


def dsinc(x):
    return np.where(x >= 0, -np.pi * jn(1, np.pi * x),
                    np.pi * jn(1, -np.pi * x))


def ddirac(m, x, x0):
    z = (m - .5) * (x - x0) / np.pi
    zz = .5 * (x - x0) / np.pi
    a = (m - .5)**2 / np.pi * dsinc(z) / np.sinc(zz)
    b = (m - .5) / 2 / np.pi * np.sinc(z) * dsinc(zz) / np.sinc(zz)**2
    c = .5 * m * np.sin(m * x)
    return (-a + b + c) / float(m)


def dvdirac(m, x, x0, v):
    a = v[0] * np.outer(ddirac(m, x, x0[0]), dirac(m, x, x0[1]))
    b = v[1] * np.outer(dirac(m, x, x0[0]), ddirac(m, x, x0[1]))
    return a + b
