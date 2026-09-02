### cahn_hilliard  (grid=24, train_steps=12, eval_steps=48, epochs=200, seeds=3)

| arch | rel_l2 | mse | psnr | conservation_err | bc_residual | grad_energy | params | train_wall_s | infer_s_per_step |
|---|---|---|---|---|---|---|---|---|---|
| bounded_cons_nca | **6.029e-01±1.2e-02** | **2.458e-01±8.2e-03** | **1.212e+01±1.5e-01** | **2.421e-04±5.4e-05** | **1.101e+00±1.0e-01** | 8.082e-01±4.7e-03 | 4576 | 6.290e+01±1.0e+00 | **6.565e-04±9.1e-05** |
| bounded_multiscale_nca | 6.329e-01±2.1e-03 | 2.708e-01±2.0e-03 | 1.169e+01±3.2e-02 | 3.612e-04±1.2e-04 | 1.141e+00±8.0e-02 | **8.150e-01±5.3e-03** | **5520** | **5.449e+01±2.7e-01** | 7.065e-04±5.8e-05 |
