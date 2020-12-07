import numpy as np

def PCG(matrix, preconditioner, rhs, tol=1e-7):
    tol = np.linalg.norm(rhs) * tol
    it = 0
    u = np.zeros_like(rhs)
    r = rhs
    if PP == None:
        z = r
    else:
        z = preconditioner * r
    p = z
    while np.sqrt(np.dot(r, r)) > tol and it < np.size(rhs):
        rz = np.dot(r, z)
        Mp = matrix * p
        alpha = rz / np.dot(p, Mp)
        u = u + alpha * p
        r = r - alpha * Mp
        if preconditioner == None:
            z = r
        else:
            z = preconditioner * r
        beta = np.dot(r, z) / rz
        p = z + beta * p
        it = it + 1
    return u, it
