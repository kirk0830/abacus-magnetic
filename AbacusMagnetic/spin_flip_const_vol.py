'''
Introduction
------------
This script is used to calculate the Potential Energy Surface (PES) during
the spin flip between ferromagnetic (FM) and antiferromagnetic (AFM) states
for a given system. A more specific context would be calculating the PBE of
the Fe2 (BCC convential cell) system, reproducing the result shown in Figure
6c of the paper

Rinaldi M, Mrovec M, Bochkarev A, et al. 
Non-collinear magnetic atomic cluster expansion for iron[J]. 
npj Computational Materials, 2024, 10(1): 12.

.
Version
-------
2024/12/21 22:04

Usage
-----
Read the following:
```python
    mags = [np.array(sph_to_cart(m, a, 0)) for m, a \
            in it.product(np.linspace(magmin, magmax, num_mag, endpoint=True), 
                          np.linspace(anglmin, anglmax, num_angl, endpoint=True))]

    pertmags = [
        [0],
        mags
    ]
```

magmin: float
    The minimum value of the magnetization magnitude, unit in uB.
magmax: float
    The maximum value of the magnetization magnitude, unit in uB.
num_mag: int
    The number of magnetization magnitudes to calculate.
anglmin: float
    The minimum value of the angle, unit in degree. Always starts from 0: FM state.
anglmax: float
    The maximum value of the angle, unit in degree. Always ends at 180: AFM state.
num_angl: int
    The number of angles to calculate.

After specifying the above parameters, run the script to generate the PES.
'''
# built-in modules
import time
import logging
import itertools as it
import argparse
import unittest

# external modules
import numpy as np
import matplotlib.pyplot as plt

# home-made modules
from deltaspin import main as dp_calculator
from utils import sph_to_cart, init

# font style as Arial
plt.rcParams['font.family'] = 'Arial'

def main(proto, overwrite, fmt, fmodel, prefix,
         magmin=1, magmax=3, magnum=31,
         angmin=0, angmax=np.pi, angnum=31,
         v=0):
    '''a hard-coded workflow to prepare the data shown in Fig.6(c) in
    Rinaldi M, Mrovec M, Bochkarev A, et al.
    Non-collinear magnetic atomic cluster expansion for iron[J].
    npj Computational Materials, 2024, 10(1): 12.
    '''
    mags = [np.array(sph_to_cart(m, a, 0)) for m, a \
            in it.product(np.linspace(magmin, magmax, magnum, endpoint=True), 
                          np.linspace(angmin, angmax, angnum, endpoint=True))]
    cell = [v]

    pertkinds = ['cell', 'magmom:atom:1']
    pertmags = [cell, mags]

    flog = f'DeltaspinPESGenerator@{time.strftime("%Y%m%d-%H%M%S")}.log'
    jobdir = f'DeltaspinPESGeneratorJob@{time.strftime("%Y%m%d-%H%M%S")}'
    logging.basicConfig(filename=flog, 
                        level=logging.INFO, 
                        format='%(asctime)s - %(message)s')
    # astonishing! cp2kdata overwrited the logging???
    logging.info(f'log file: {flog}')
    logging.info(f'generated jobs will be in folder: {jobdir}')

    dp_calculator(fn=proto, 
                  fmt=fmt, 
                  pertkinds=pertkinds,
                  pertmags=pertmags, 
                  jobdir=jobdir,
                  overwrite=overwrite,
                  fmodel=fmodel,
                  prefix=prefix)

    logging.info('done')
    logging.shutdown()
    
    return jobdir

class TestSpinFlipConstVolume(unittest.TestCase):
    
    @unittest.skip('NOTHING TO TEST')
    def test_nothing(self):
        self.assertEqual(1, 1)

if __name__ == '__main__':
    
    test = init()
    unittest.main(exit=test)
    
    _ = main(
        # control
        # -------
        prefix='Fig6c',
        
        # basic structure definition
        # --------------------------
        proto='BCC-Fe-primitive',
        fmt='abacus/stru',
        
        # magmom of other atoms
        # ---------------------
        overwrite=2.4,           

        # energy calculator
        # -----------------
        fmodel='/share/pxl/Fe-dp-model/lcao-all-dpa1-20241219-protection0.02/frozen_model.pth',

        # magmom scan
        # -----------
        magmin=1, magmax=3,     magnum=31,
        angmin=0, angmax=np.pi, angnum=31,
        
        # cell volume variation
        # ----------------------
        v=0
        )
    
    axis_font_size = 12

    angles = np.linspace(0, 180, 31)
    magnorm = np.linspace(1, 3, 31, endpoint=True)
    magmom1 = np.loadtxt('Fig6c.e_peratom.out', skiprows=1)[:,1].reshape(31, 31)
    #magmom2 = np.loadtxt('magmom-2.e_peratom.out', skiprows=1)[:,1]
    magmom1 = magmom1 - np.min(magmom1) - 7.88

    fig, ax = plt.subplots(figsize=(5, 6))
    # use jet color map
    colors = plt.cm.jet(np.linspace(0, 1, 31))
    for i in range(31):
        ax.plot(angles, magmom1[i], color=colors[i])
    ax.set_xlabel('$\\theta$ (degree)', fontsize=axis_font_size)
    ax.set_ylabel('Energy (eV/atom)', fontsize=axis_font_size)
    ax.set_ylim(-7.9, -7.25)

    # Adjust the colorbar to have a scale from 1 to 3 with 5 ticks
    norm = plt.Normalize(vmin=1, vmax=3)  # Set the normalization from 1 to 3
    sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])  # You need to set the array for the ScalarMappable

    # color bar
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', ticks=[1, 1.5, 2, 2.5, 3])
    cbar.set_label('Moment ($\\mu_B$/atom)', fontsize=axis_font_size)
    cbar.ax.set_xticklabels(['1', '1.5', '2', '2.5', '3'])  # Set the tick labels

    plt.tight_layout()
    plt.savefig('fig6c.png', dpi=300)
