### Numerical dispersion of the wave solver (grid=48)

The continuum 2-D wave equation is **non-dispersive**: every Fourier mode travels at the same speed $c=0.5$. The scheme the benchmark distils from is not. For each single-mode initial condition the table gives the exact frequency $c|k|$, the semi-discrete frequency implied by the 5-point Laplacian, the fully discrete frequency of the symplectic stepper, and the frequency fitted from the solver's own trajectory.

| mode $(m_x,m_y)$ | $\|k\|$ | $\omega$ exact | $\omega$ semi-discrete | $\omega$ full discrete | $\omega$ measured | phase error | measured phase speed |
|---|---|---|---|---|---|---|---|
| (1,0) | 0.131 | 0.0654 | 0.0654 | 0.0654 | 0.0654 | -0.06% | 0.4997 |
| (2,0) | 0.262 | 0.1309 | 0.1305 | 0.1305 | 0.1305 | -0.28% | 0.4986 |
| (4,0) | 0.524 | 0.2618 | 0.2588 | 0.2588 | 0.2588 | -1.14% | 0.4943 |
| (8,0) | 1.047 | 0.5236 | 0.5000 | 0.5000 | 0.5000 | -4.50% | 0.4775 |
| (12,0) | 1.571 | 0.7854 | 0.7071 | 0.7071 | 0.7071 | -9.96% | 0.4502 |
| (16,0) | 2.094 | 1.0472 | 0.8660 | 0.8661 | 0.8661 | -17.29% | 0.4135 |
| (20,0) | 2.618 | 1.3090 | 0.9659 | 0.9660 | 0.9660 | -26.20% | 0.3690 |
| (24,0) | 3.142 | 1.5708 | 1.0000 | 1.0001 | 1.0001 | -36.33% | 0.3183 |
| (1,1) | 0.185 | 0.0926 | 0.0925 | 0.0925 | 0.0925 | -0.08% | 0.4996 |
| (2,2) | 0.370 | 0.1851 | 0.1846 | 0.1846 | 0.1846 | -0.29% | 0.4986 |
| (4,4) | 0.740 | 0.3702 | 0.3660 | 0.3660 | 0.3660 | -1.14% | 0.4943 |
| (8,8) | 1.481 | 0.7405 | 0.7071 | 0.7071 | 0.7071 | -4.50% | 0.4775 |
| (12,12) | 2.221 | 1.1107 | 1.0000 | 1.0001 | 1.0001 | -9.96% | 0.4502 |
| (16,16) | 2.962 | 1.4810 | 1.2247 | 1.2249 | 1.2249 | -17.29% | 0.4136 |
| (24,24) | 4.443 | 2.2214 | 1.4142 | 1.4145 | 1.4145 | -36.32% | 0.3184 |

The fitted frequencies track the fully discrete prediction, and the phase error grows monotonically with $|k|$: from 0.06% at mode $(1,0)$ to 36.3% at $(24,0)$, where the measured phase speed is 0.318 against the exact 0.5. Both error sources push the same way, so short wavelengths travel too slowly.

The honest description of the phenomenon is therefore that the **continuum equation is non-dispersive and its discretisation is dispersive at the grid scale**, which is also why a model that resolves only smooth structure can still score well on it.
