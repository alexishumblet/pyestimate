# Author: Alexis Humblet
# 2024

from estimate.estimators import sin_param_estimate

import numpy as np

def test_sin_param_estimate_no_fft():
    N = 100
    n = np.arange(N)
    sigma = 0.1
    w = np.random.default_rng(seed=0).normal(scale=sigma, size=N) # white gaussian noise

    # use_fft = False test case
    for f in np.linspace(1/N, 0.5-1/N, 5):
        for A in np.linspace(0.1, 10, 5):
            for phi in np.linspace(-np.pi, np.pi, 5):
                x = A * np.cos(2*np.pi*f*n+phi) + w
                A_hat, f_hat, phi_hat = sin_param_estimate(x, use_fft=False, brute_Ns=1000)

                A_hat_std = np.sqrt(2*sigma**2/N)
                eta = A**2/2/sigma**2
                phi_hat_std = np.sqrt(2*(2*N-1)/eta/N/(N+1))
                f_hat_std = np.sqrt(12/(2*np.pi)**2/eta/N/(N**2-1))
                phi_err = phi_hat-phi
                if phi_err > np.pi:
                    phi_err -= 2*np.pi
                elif phi_err < -np.pi:
                    phi_err += 2*np.pi

                assert np.abs(A_hat-A) < 3*A_hat_std
                assert np.abs(phi_err) < 3*phi_hat_std
                assert np.abs(f_hat-f) < 3*f_hat_std

    return

def test_sin_param_estimate_fft():
    N = 100
    n = np.arange(N)
    sigma = 0.5
    w = np.random.default_rng(seed=0).normal(scale=sigma, size=N) # white gaussian noise

    # use_fft = True test case
    for f in np.linspace(8/N, 0.5-8/N, 5):
        for A in np.linspace(0.5, 10, 5):
            for phi in np.linspace(-np.pi, np.pi, 5):
                x = A * np.cos(2*np.pi*f*n+phi) + w
                A_hat, f_hat, phi_hat = sin_param_estimate(x, use_fft=True, nfft=2**14)

                A_hat_std = np.sqrt(2*sigma**2/N)
                eta = A**2/2/sigma**2
                phi_hat_std = np.sqrt(2*(2*N-1)/eta/N/(N+1))
                f_hat_std = np.sqrt(12/(2*np.pi)**2/eta/N/(N**2-1))
                phi_err = phi_hat-phi
                if phi_err > np.pi:
                    phi_err -= 2*np.pi
                elif phi_err < -np.pi:
                    phi_err += 2*np.pi

                assert np.abs(f_hat-f) < 6*f_hat_std          
                assert np.abs(A_hat-A) < 6*A_hat_std
                assert np.abs(phi_err) < 6*phi_hat_std
                
    return

def test_sin_param_estimate_known_f():
    N = 100
    n = np.arange(N)
    sigma = 0.1
    w = np.random.default_rng(seed=0).normal(scale=sigma, size=N) # white gaussian noise

    for f in np.linspace(1/N, 0.5-1/N, 50):
        for A in np.linspace(0.1, 10, 50):
            for phi in np.linspace(-np.pi, np.pi, 50):
                x = A * np.cos(2*np.pi*f*n+phi) + w
                A_hat, _, phi_hat = sin_param_estimate(x, freq=f)

                A_hat_std = np.sqrt(2*sigma**2/N)
                eta = A**2/2/sigma**2
                phi_hat_std = np.sqrt(2*(2*N-1)/eta/N/(N+1))
                phi_err = phi_hat-phi
                if phi_err > np.pi:
                    phi_err -= 2*np.pi
                elif phi_err < -np.pi:
                    phi_err += 2*np.pi

                assert np.abs(A_hat-A) < 3*A_hat_std
                assert np.abs(phi_err) < 3*phi_hat_std

    return