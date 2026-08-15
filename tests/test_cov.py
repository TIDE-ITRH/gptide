import numpy as np

from gptide.cov import (
    expquad_1d,
    matern12_1d,
    matern32_1d,
    matern52_1d,
    cosine_1d,
    periodic_1d,
    matern_general_1d,
    matern32_matrix,
)


def test_kernels_return_variance_at_zero_lag():
    x = np.array([0.0])
    eta, l = 2.0, 3.0

    assert expquad_1d(x, x, (eta, l))[0] == eta**2
    assert matern12_1d(x, x, (eta, l))[0] == eta**2
    assert matern32_1d(x, x, (eta, l))[0] == eta**2
    assert matern52_1d(x, x, (eta, l))[0] == eta**2
    assert cosine_1d(x, x, (eta, l))[0] == eta**2


def test_kernels_decay_with_distance():
    xpr = np.array([0.0])
    eta, l = 1.0, 1.0

    near = np.array([0.1])
    far = np.array([5.0])

    for kernel in (expquad_1d, matern12_1d, matern32_1d, matern52_1d):
        k_near = kernel(near, xpr, (eta, l))[0]
        k_far = kernel(far, xpr, (eta, l))[0]
        assert k_near < eta**2
        assert k_far < k_near


def test_periodic_kernel_repeats_with_period():
    eta, l, p = 1.0, 1.0, 2.0
    xpr = np.array([0.0])

    k0 = periodic_1d(np.array([0.0]), xpr, (eta, l, p))[0]
    k_period = periodic_1d(np.array([p]), xpr, (eta, l, p))[0]

    np.testing.assert_allclose(k0, k_period)
    np.testing.assert_allclose(k0, eta**2)


def test_matern_general_matches_matern12_at_nu_half():
    x = np.array([0.5, 1.5, 3.0])
    xpr = np.array([0.0])
    eta, l = 1.5, 2.0

    general = matern_general_1d(x, xpr, (eta, 0.5, l))
    reference = matern12_1d(x, xpr, (eta, l))

    np.testing.assert_allclose(general, reference, rtol=1e-6)


def test_matern32_matrix_is_symmetric_positive_definite():
    x = np.array([0.0, 1.0, 2.5, 4.0])
    d = np.abs(x[:, None] - x[None, :])
    K = matern32_matrix(d, l=2.0)

    np.testing.assert_allclose(K, K.T)
    eigvals = np.linalg.eigvalsh(K)
    assert np.all(eigvals > 0)
