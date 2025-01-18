run phonon.py with
```python
    out = main(relaxed='phonopy-Diamond-Si/Diamond-Si-primitive-lcao',
               calculator='mpirun -np 16 abacus',
               phonopy_dim=[2, 2, 2],
               fdft='phonopy-Diamond-Si/INPUT',
               fpsp='phonopy-Diamond-Si/Si_ONCV_PBE-1.0.upf',
               forb='phonopy-Diamond-Si/Si_gga_10au_100Ry_2s2p1d.orb')
```

Note: for test, diamond q-points should be
https://journals.aps.org/prb/pdf/10.1103/PhysRevB.74.054302
Gamma-X-K-Gamma-L
0 0 0  0.5 0 0.5  0.375 0.375 0.75  0 0 0  0.5 0.5 0.5