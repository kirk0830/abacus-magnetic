'''
Introduction
------------
This script is used to calculate the Potential Energy Surface (PES) during
the spin flip between ferromagnetic (FM) and antiferromagnetic (AFM) states
for a given system. A more specific context would be calculating the PBE of
the Fe2 (BCC convential cell) system, reproducing the result shown in Figure
6b of the paper

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
        np.linspace(vmin, vmax, num_v, endpoint=True),
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
vmin: float
    The minimum value of the volume scale, unit in %.
vmax: float
    The maximum value of the volume scale, unit in %.
num_v: int
    The number of volume scales to calculate.

After specifying the above parameters, run the script to generate the PES.
'''
# built-in modules
import time
import logging
import itertools as it
import unittest

# external modules
import numpy as np
import matplotlib.pyplot as plt

# home-made modules
from deltaspin import main as dp_calculator
from utils import sph_to_cart, cart_to_sph, init

# font style as Arial
plt.rcParams['font.family'] = 'Arial'

def cal_vol(cell):
    '''calculate the volume of the cell
    
    Parameters
    ----------
    cell : np.array
        a 3x3 or 9-element array representing the cell vectors
    '''
    if not isinstance(cell, np.ndarray):
        raise ValueError('cell must be a np.array')
    cell = cell.reshape(3, 3)
    v = np.linalg.det(cell)
    if v == 0:
        raise ValueError(f'zero volume: {cell}')
    return v

def read_box(fn):
    '''read the cell information stored in an npy file'''
    cell = np.load(fn)
    return cell

def cal_mag(mag):
    '''
    calculate the magnetization r, theta, phi components
    for each site
    '''
    mag = mag.reshape(-1, 3)
    return [cart_to_sph(*m, deg=True) for m in mag]

def read_mag(fn):
    '''read the magnetization stored in an npy file'''
    mag = np.load(fn)
    return mag

def parse(fbox, fmag, iat, fener):
    '''combine the information of box size, atomic mags
    and energies together'''
    fbox = [fbox] if isinstance(fbox, str) else fbox
    cell = np.concatenate([read_box(f) for f in fbox])
    
    fmag = [fmag] if isinstance(fmag, str) else fmag
    mag = np.concatenate([read_mag(f) for f in fmag])

    ener = np.loadtxt(fener, skiprows=1)[:,1] # the second column
    
    if not (len(cell) == len(mag) == len(ener)):
        raise ValueError(f'inconsistent length: {len(cell)}, {len(mag)}, {len(ener)}')
    
    return [(round(cal_vol(c), 2), round(cal_mag(m)[iat][1], 2), e) for c, m, e in zip(cell, mag, ener)]

def vol_theta_lowest_e(data):
    '''find the lowest energy for each (volume, theta) pair'''
    print(f'Postprocessing, data has dimension {len(data)}')
    # print(data)
    vol_theta_e = {}
    for vol, theta, e in data:
        if (vol, theta) in vol_theta_e:
            if e <= vol_theta_e[(vol, theta)]:
                vol_theta_e[(vol, theta)] = e
        else:
            vol_theta_e[(vol, theta)] = e
    print(f'After postprocessing, data has dimension {len(vol_theta_e)}')
    # sort the keys first by volume, then by theta
    keys = sorted(vol_theta_e.keys(), key=lambda x: (x[0], x[1]))

    nvol = len(set([k[0] for k in keys]))
    ntheta = len(set([k[1] for k in keys]))
    # only keep the values
    return np.array([vol_theta_e[k] for k in keys]).reshape(nvol, ntheta)

def plot(data, x, y):

    fontsize = 12
    ener = vol_theta_lowest_e(data).T
    ener -= ener.max()

    cmap = plt.get_cmap('Spectral')

    fig, ax = plt.subplots(figsize=(5, 6))
    ax.imshow(ener, origin='lower', aspect='auto', cmap=cmap, interpolation='nearest')
    ax.contour(ener, levels=10, colors='black', origin='lower', linewidths=0.5)
    # ax.contourf(ener, levels=10, cmap=cmap, origin='lower')
    
    ax.set_xlabel(f'{x["label"]} ({x["unit"]})', fontsize=fontsize)
    ax.set_xticks(np.linspace(0, ener.shape[1], 5))
    ax.set_xticklabels([f'{xi:.2f}' for xi in np.linspace(x['min'], x['max'], 5, endpoint=True)])
    ax.set_xlim(1, ener.shape[1]-1)

    ax.set_ylabel(f'{y["label"]} ({y["unit"]})', fontsize=fontsize)
    ax.set_yticks(np.linspace(0, ener.shape[0], 10))
    ax.set_yticklabels(np.linspace(y['min'], y['max'], 10, endpoint=True).astype(int))
    ax.set_ylim(1, ener.shape[0]-1)

    # colormap normalization
    norm = plt.Normalize(vmin=ener.min(), vmax=ener.max())

    # create scalar mappable based on normalized colormap
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Create a colorbar for the ScalarMappable instance
    cbar = ax.figure.colorbar(mappable, 
                              ax=ax, 
                              orientation='horizontal', 
                              ticks=np.linspace(ener.min(), ener.max(), 5))

    # format the colorbar labels and title
    cbar.set_label('Energy (eV/atom)', fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    plt.tight_layout()
    plt.savefig(f'fig6b.png', dpi=300, bbox_inches='tight')

def main(proto, overwrite, fmt, fmodel, prefix,
         magmin=1, magmax=3, magnum=31,
         angmin=0, angmax=np.pi, angnum=31,
         vmin=-0.3, vmax=0.3, vnum=31):
    '''a hard-coded workflow to prepare the data shown in Fig.6(b) in
    Rinaldi M, Mrovec M, Bochkarev A, et al. 
    Non-collinear magnetic atomic cluster expansion for iron[J]. 
    npj Computational Materials, 2024, 10(1): 12.
    '''
    mags = [np.array(sph_to_cart(m, a, 0)) for m, a \
            in it.product(np.linspace(magmin, magmax, magnum, endpoint=True), 
                          np.linspace(angmin, angmax, angnum, endpoint=True))]
    cell = np.linspace(vmin, vmax, vnum, endpoint=True)

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

class TestSpinFilpScanVolume(unittest.TestCase):
    
    @unittest.skip('NOTHING TO TEST')
    def test_nothing(self):
        self.assertEqual(1, 1)

if __name__ == '__main__':

    test = init()
    unittest.main(exit=test)
    
    root = main(
        # control
        # -------
        prefix='Fig6b',
        
        # basic structure definition
        # --------------------------
        proto='BCC-Fe-primitive',
        fmt='abacus/stru',
        
        # magmom scan
        # -----------
        magmin=1, magmax=3,     magnum=31,
        angmin=0, angmax=np.pi, angnum=31,
        
        # cell volume scan
        # ----------------
        vmin=-0.3, vmax=0.3, vnum=31,
        
        # magmom of other atoms
        # ---------------------
        overwrite=2.4,

        # energy calculator
        # -----------------
        fmodel='/share/pxl/Fe-dp-model/lcao-all-dpa1-20241219-protection0.02/frozen_model.pth',
        )
    
    # postprocessing
    data = parse(fbox = [f'{root}/Fe2/set.000/box.npy',
                         f'{root}/Fe2/set.001/box.npy',
                         f'{root}/Fe2/set.002/box.npy',
                         f'{root}/Fe2/set.003/box.npy',
                         f'{root}/Fe2/set.004/box.npy',
                         f'{root}/Fe2/set.005/box.npy'], 
                 fmag = [f'{root}/Fe2/set.000/spin.npy',
                         f'{root}/Fe2/set.001/spin.npy',
                         f'{root}/Fe2/set.002/spin.npy',
                         f'{root}/Fe2/set.003/spin.npy',
                         f'{root}/Fe2/set.004/spin.npy',
                         f'{root}/Fe2/set.005/spin.npy'],
                 iat = 0,
                 fener = 'Fig6b.e_peratom.out')
    plot(data, 
         x={'label': 'Volume scale', 'unit': '%', 'min': -30, 'max': 30},
         y={'label': '$\\theta$', 'unit': 'degree', 'min': 0, 'max': 180})
