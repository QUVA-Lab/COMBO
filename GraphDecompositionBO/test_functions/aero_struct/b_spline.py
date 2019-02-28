from __future__ import division
import numpy
import scipy.sparse


def get_bspline_mtx(num_cp, num_pt, order=4):
    """ Create Jacobian to fit a bspline to a set of data.

    Parameters
    ----------
    num_cp : int
        Number of control points.
    num_pt : int
        Number of points.
    order : int, optional
        Order of b-spline fit.

    Returns
    -------
    out : CSR sparse matrix
        Matrix that gives the points vector when multiplied by the control
        points vector.

    """
    knots = numpy.zeros(num_cp + order)
    knots[order-1:num_cp+1] = numpy.linspace(0, 1, num_cp - order + 2)
    knots[num_cp+1:] = 1.0
    t_vec = numpy.linspace(0, 1, num_pt)

    basis = numpy.zeros(order)
    arange = numpy.arange(order)
    data = numpy.zeros((num_pt, order))
    rows = numpy.zeros((num_pt, order), int)
    cols = numpy.zeros((num_pt, order), int)

    for ipt in xrange(num_pt):
        t = t_vec[ipt]

        i0 = -1
        for ind in xrange(order, num_cp+1):
            if (knots[ind-1] <= t) and (t < knots[ind]):
                i0 = ind - order
        if t == knots[-1]:
            i0 = num_cp - order

        basis[:] = 0.
        basis[-1] = 1.

        for i in xrange(2, order+1):
            l = i - 1
            j1 = order - l
            j2 = order
            n = i0 + j1
            if knots[n+l] != knots[n]:
                basis[j1-1] = (knots[n+l] - t) / \
                              (knots[n+l] - knots[n]) * basis[j1]
            else:
                basis[j1-1] = 0.
            for j in range(j1+1, j2):
                n = i0 + j
                if knots[n+l-1] != knots[n-1]:
                    basis[j-1] = (t - knots[n-1]) / \
                                (knots[n+l-1] - knots[n-1]) * basis[j-1]
                else:
                    basis[j-1] = 0.
                if knots[n+l] != knots[n]:
                    basis[j-1] += (knots[n+l] - t) / \
                                  (knots[n+l] - knots[n]) * basis[j]
            n = i0 + j2
            if knots[n+l-1] != knots[n-1]:
                basis[j2-1] = (t - knots[n-1]) / \
                              (knots[n+l-1] - knots[n-1]) * basis[j2-1]
            else:
                basis[j2-1] = 0.

        data[ipt, :] = basis
        rows[ipt, :] = ipt
        cols[ipt, :] = i0 + arange

    data, rows, cols = data.flatten(), rows.flatten(), cols.flatten()

    return scipy.sparse.csr_matrix((data, (rows, cols)),
                                   shape=(num_pt, num_cp))

if __name__ == "__main__":


    num_cp = 5
    num_pt = 100
    rng = 5 # 700 * 1.852 / 1e3
    alt = 11

    lins = numpy.linspace(0, 1, num_cp)
    cos_dist = 0.5 * (1 - numpy.cos(lins * numpy.pi))
    x_cp = rng * cos_dist
    h_cp = alt * numpy.sin(numpy.pi * cos_dist)

    import time
    t0 = time.time()
    jac = get_bspline_mtx(num_cp, num_pt)
    print(time.time() - t0)

    h_cp[3] += 2

    x = jac.dot(x_cp)
    h = jac.dot(h_cp)

    import pylab
    pylab.plot(x_cp, h_cp, 'o')
    pylab.plot(x, h)
    pylab.show()
