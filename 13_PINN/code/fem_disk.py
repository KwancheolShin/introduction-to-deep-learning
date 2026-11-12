# Helper (no need to read it): reference solutions by the finite element method (P1 triangles) on the unit disk,
#   -div(sigma grad u) = 0  with  u = f  (solve_dirichlet)  or  sigma du/dn = g  (solve_neumann)  on the boundary.
# Used by pinn_elliptic_2d.py (reference solution) and pinn_eit.py (synthetic boundary data).
# The error is about 1e-5 for h = 0.01 (checked against r^n cos(n theta) / n for sigma = 1; run this file).
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import Delaunay

def disk_mesh(h=0.02):
    pts = [np.zeros((1, 2))]
    n_rings = int(round(1 / h))
    for i in range(1, n_rings + 1):
        r = i / n_rings
        m = max(6, int(round(2 * np.pi * r / h)))
        th = 2 * np.pi * np.arange(m) / m + (0.5 * i) * (2 * np.pi / m)   # stagger rings
        pts.append(np.c_[r * np.cos(th), r * np.sin(th)])
    p = np.vstack(pts)
    t = Delaunay(p).simplices
    nb = m                                  # boundary nodes are the last ring
    bnd = np.arange(len(p) - nb, len(p))
    return p, t, bnd

def assemble(p, t, sigma):
    x = p[t]                                # (T, 3, 2)
    d1 = x[:, 1] - x[:, 0]; d2 = x[:, 2] - x[:, 0]
    area = 0.5 * np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    # gradients of the barycentric functions
    B = np.stack([np.c_[x[:, 1, 1] - x[:, 2, 1], x[:, 2, 0] - x[:, 1, 0]],
                  np.c_[x[:, 2, 1] - x[:, 0, 1], x[:, 0, 0] - x[:, 2, 0]],
                  np.c_[x[:, 0, 1] - x[:, 1, 1], x[:, 1, 0] - x[:, 0, 0]]], axis=1) / (2 * area)[:, None, None]
    c = x.mean(axis=1)
    s = sigma(c[:, 0], c[:, 1])
    Ke = (s * area)[:, None, None] * np.einsum("tik,tjk->tij", B, B)
    I = np.repeat(t, 3, axis=1).ravel(); J = np.tile(t, (1, 3)).ravel()
    return sp.csr_matrix((Ke.ravel(), (I, J)), shape=(len(p), len(p))).tocsc()

def boundary_load(p, bnd, g):
    th = np.arctan2(p[bnd, 1], p[bnd, 0])
    m = len(bnd)
    b = np.zeros(len(p))
    for i in range(m):
        a, c = bnd[i], bnd[(i + 1) % m]
        L = np.linalg.norm(p[a] - p[c])
        tm = np.arctan2(*(0.5 * (p[a] + p[c]))[::-1])
        ga, gc, gm = g(th[i]), g(th[(i + 1) % m]), g(tm)
        b[a] += L * (ga / 6 + gm / 3)       # Simpson for int g phi_a
        b[c] += L * (gc / 6 + gm / 3)
    return b

def solve_neumann(p, t, bnd, sigma, g, K=None):
    if K is None:
        K = assemble(p, t, sigma)
    b = boundary_load(p, bnd, g)
    n = len(p)
    # mean-zero constraint on the boundary via a Lagrange multiplier
    c = np.zeros(n); c[bnd] = 1.0 / len(bnd)
    A = sp.bmat([[K, sp.csc_matrix(c[:, None])], [sp.csc_matrix(c[None, :]), None]]).tocsc()
    u = spla.spsolve(A, np.r_[b, 0.0])[:n]
    return u

def solve_dirichlet(p, t, bnd, sigma, f_bnd, K=None):
    if K is None:
        K = assemble(p, t, sigma)
    n = len(p)
    inner = np.setdiff1d(np.arange(n), bnd)
    u = np.zeros(n); u[bnd] = f_bnd
    rhs = -K[inner][:, bnd] @ f_bnd
    u[inner] = spla.spsolve(K[inner][:, inner], rhs)
    return u

if __name__ == "__main__":
    import time
    for h in [0.04, 0.02, 0.01]:
        t0 = time.time()
        p, t, bnd = disk_mesh(h)
        one = lambda x, y: np.ones_like(x)
        errs = []
        for n in [1, 2]:
            u = solve_neumann(p, t, bnd, one, lambda th: np.cos(n * th))
            r = np.hypot(p[:, 0], p[:, 1]); th = np.arctan2(p[:, 1], p[:, 0])
            ue = r**n * np.cos(n * th) / n
            errs.append(np.abs(u - ue).max())
        print(f"h={h}: nodes {len(p)}, max err n=1 {errs[0]:.2e}, n=2 {errs[1]:.2e}  ({time.time()-t0:.1f}s)")
