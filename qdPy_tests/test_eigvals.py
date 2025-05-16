import numpy as np
import jax.scipy.linalg 
import scipy.sparse
from numpy.testing import assert_array_almost_equal

def test_eigvals():
    # constructing the matrix
    mat_sparse = scipy.sparse.rand(100,100)
    mat_dense = mat_sparse.todense()

    # making symmetric matrix
    sym_mat = mat_dense + mat_dense.T

    # calculating the eigenvalues
    w_np, __ = np.linalg.eigh(sym_mat)
    w_jx, __ = jax.scipy.linalg.eigh(sym_mat)

    # checking if equal
    assert_array_almost_equal(w_np, w_jx)
