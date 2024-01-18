# Author: Alexis Humblet
# 2024

from pyestimate import sin2d_param_estimate

import numpy as np

def test_sin2d_param_estimate_no_fft():
    N = 10
    M = 15
    NM = N*M
    n, m = np.meshgrid(np.arange(N), np.arange(M), indexing='ij')
    sigma = np.sqrt(0.2)
    w = np.random.default_rng(seed=0).normal(scale=sigma, size=(N,M)) # white gaussian noise

     # use_fft = False test case
    for fn in np.linspace(1/N, 0.5-1/N, 3):
        for fm in np.linspace(1/N, 0.5-1/N, 3):
            for A in np.linspace(0.1, 10, 3):
                for phi in np.linspace(-np.pi, np.pi, 3):
                    x = A * np.cos(2*np.pi*(fn*n+fm*m)+phi) + w
                    A_hat, f_hat, phi_hat = sin2d_param_estimate(x, use_fft=False, brute_Ns=100)

                A_hat_std = np.sqrt(2*sigma**2/NM)
                eta = A**2/2/sigma**2
                phi_hat_std = np.sqrt(2*(2*NM-1)/eta/NM/(NM+1))
                f_n_hat_std = np.sqrt(12/(2*np.pi)**2/eta/N/(N**2-1))
                f_m_hat_std = np.sqrt(12/(2*np.pi)**2/eta/M/(M**2-1))
                phi_err = phi_hat-phi
                if phi_err > np.pi:
                    phi_err -= 2*np.pi
                elif phi_err < -np.pi:
                    phi_err += 2*np.pi

                assert np.abs(A_hat-A) < 3*A_hat_std
                assert np.abs(phi_err) < 3*phi_hat_std
                assert np.abs(f_hat[0]-fn) < 4*f_n_hat_std
                assert np.abs(f_hat[1]-fm) < 4*f_m_hat_std

    return

def test_sin2d_param_estimate_fft():
    N = 30
    M = 40
    NM = N*M
    n, m = np.meshgrid(np.arange(N), np.arange(M), indexing='ij')
    sigma = np.sqrt(0.2)
    w = np.random.default_rng(seed=0).normal(scale=sigma, size=(N,M)) # white gaussian noise

     # use_fft = True test case
    for fn in np.linspace(8/N, 0.5-8/N, 3):
        for fm in np.linspace(8/M, 0.5-8/M, 3):
            for A in np.linspace(0.1, 10, 3):
                for phi in np.linspace(-np.pi, np.pi, 3):
                    x = A * np.cos(2*np.pi*(fn*n+fm*m)+phi) + w
                    A_hat, f_hat, phi_hat = sin2d_param_estimate(x, use_fft=True, nfft=[4096, 4096])

                A_hat_std = np.sqrt(2*sigma**2/NM)
                eta = A**2/2/sigma**2
                phi_hat_std = np.sqrt(2*(2*NM-1)/eta/NM/(NM+1))
                f_n_hat_std = np.sqrt(12/(2*np.pi)**2/eta/N/(N**2-1))
                f_m_hat_std = np.sqrt(12/(2*np.pi)**2/eta/M/(M**2-1))
                phi_err = phi_hat-phi
                if phi_err > np.pi:
                    phi_err -= 2*np.pi
                elif phi_err < -np.pi:
                    phi_err += 2*np.pi

                assert np.abs(A_hat-A) < 6*A_hat_std
                assert np.abs(phi_err) < 6*phi_hat_std
                assert np.abs(f_hat[0]-fn) < 6*f_n_hat_std
                assert np.abs(f_hat[1]-fm) < 6*f_m_hat_std

    return
