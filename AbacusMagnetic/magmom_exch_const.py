'''
In brief
--------
Employing the small angle approximation, calculate the distance-
dependent magnetic exchange coefficient Jij(r) for a pair of atoms.


Formulation
-----------
  dEij(thetaij) - (dEi(thetaij) + dEj(thetaij)) 
= Jij(r) * (1 - cos(thetaij))
~ Jij(r) * (thetaij)^2 / 2

, where all dEx terms are the energy difference between the 
unperturbated and perturbated states.
For dEij, rotate the spin of two atoms spaced by r by thetaij/2
in the opposite direction.
For dEi, rotate the spin of atom i by thetaij, and for dEj,
rotate the spin of atom j by thetaij.

Varying thetaij in a small range, we can obtain the Jij(r)
by linear fitting the relation between dEij(thetaij) and
(thetaij)^2 / 2.


Usage
-----
see the code piece at the end of this file enclosed by
`if __name__ == '__main__':`


References
----------
[1] Liechtenstein A I, Katsnelson M I, Antropov V P, et al. 
    Local spin density functional approach to the theory of 
    exchange interactions in ferromagnetic metals and alloys[J]. 
    Journal of Magnetism and Magnetic Materials, 1987, 67(1): 65-74.
[2] Rinaldi M, Mrovec M, Bochkarev A, et al. 
    Non-collinear magnetic atomic cluster expansion for iron[J]. 
    npj Computational Materials, 2024, 10(1): 12.
'''
# built-in modules
import time
import logging
import os
import unittest
import shutil

# external modules
import numpy as np
import matplotlib.pyplot as plt

# home-made modules
from utils import cart_to_sph, sph_to_cart, convert_length_unit, init
from structure import supercell_count, make_shell_spikes
from deltaspin import main as _pert_kernel_impl
from deltaspin import predict as _predict_dp_impl
from deltaspin import _deepmd_signature, to_abacus, is_dpdata_dir
from deltaspin import write_energies as write_dp_energies
from deltaspin import write_forces as write_dp_forces
from deltaspin import write_virials as write_dp_virials
from abacus import read_stru, read_fdft, isfinished
from abacus import write_cell as write_abacus_stru
from abacus import read_natoms as read_natoms_from_abacus_log
from abacus import read_final_energy as read_abacus_scf_energy
from abacus import read_forces as read_abacus_forces
from abacus import build_supercell as build_abacus_supercell
# font style as Arial
plt.rcParams['font.family'] = 'Arial'

def _read_e_theta_curve(fspn, fener, iat, fcoord = None):
    '''
    read the energy-theta curve from different output files.
    The theta is stored in .npy format, so can be read directly.
    The energy is stored in .out plain text format, np.loadtxt
    with skiprows=1 is used to read the file.
    
    this function will also check the linearity of the energy-
    theta curve by a simple linear fitting, return the Pearson
    correlation coefficient.

    Additionaly, the coordinates of the atoms can be read from
    fcoord, it is also a .npy file.
    '''
    iat = [iat] if isinstance(iat, int) else iat
    
    spins = np.load(fspn)
    # stored in [i][x1, y1, z1, x2, y2, z2, ...]
    # i is the index of structure, x1, y1, z1 are the magmom of the
    # first atom, and so on.
    nframe = len(spins)
    spins = spins.reshape(nframe, -1, 3)
    _, nat, _ = spins.shape
    # calculate the angle between rotated atoms and those not rotated
    iat0 = [i for i in range(nat) if i not in iat][0]
    spin_env = spins[0, iat0, :] # the atom not rotated
    # then
    spins = spins[:, iat, :] # only the selected atoms
    theta = np.array([[np.arccos(np.dot(spin_env, spin_at)\
        / (np.linalg.norm(spin_env) * np.linalg.norm(spin_at)))\
        * 180 / np.pi
                       for spin_at in frame] for frame in spins])

    ener = np.loadtxt(fener, skiprows=1)[:,1] # the second column
    ener = ener.reshape(-1, 1)
    if len(ener) != nframe:
        raise ValueError(f'inconsistent length: {nframe}, {len(ener)}')
    
    ener -= ener[0]
    # check the linearity of the energy-theta curve for each
    # selected atom
    r = [np.corrcoef(theta[:,i]**2 / 2, ener[:,0])[0, 1] 
         for i in range(len(iat))]

    dist = None
    if fcoord is not None:
        coords = np.load(fcoord)
        coords = coords.reshape(nframe, -1, 3) # different frames
        coords = coords[:, iat]
        if len(iat) == 2: # a pair of atoms, calculate the dist
            dist = np.linalg.norm(coords[:,0] - coords[:,1], axis=1)
            print(f'fcoord is given, additionally calculate the distance between the atoms.')
            print(f'for atom-pair {iat} the distance between the atoms is:\n{dist}')

    return theta, ener.reshape(-1, 1), r, dist

def _plot_e_theta_curve(theta, ener, r, iat, ax):
    '''
    plot the energy-theta curve for selected atoms.
    '''
    iat = [iat] if isinstance(iat, int) else iat
    if len(iat) != len(r):
        raise ValueError(f'inconsistent length between iat and r: {len(iat)}, {len(r)}')
    colors = plt.cm.jet(np.linspace(0, 1, len(iat)))
    for i, c in zip(iat, colors):
        ax.plot(theta[:,i]**2/2, ener, color=c,
                marker='o', linestyle='-', markersize=2,
                label=f'atom {i} ($R^2$={r[i]:.2f})')
    ax.legend()
    ax.set_xlabel('$\\theta^2/2$ (degree^2)')
    ax.set_ylabel('Energy (eV/atom)')
    return ax

def parse(fspn, fener, iat, fcoord = None):
    '''
    read the energy-theta curve from jobs of only atom(i), only
    atom(j), and both atom(i) and atom(j) perturbated.
    '''
    # sanity check
    if len(fspn) != 3:
        raise ValueError('fspn should have length 3')
    if len(fener) != 3:
        raise ValueError('fener should have length 3')
    if len(iat) != 2:
        raise ValueError('iat should have length 2')
    
    theta_i, ener_i, R2_i, _ = _read_e_theta_curve(fspn[0], fener[0], iat[0])
    theta_j, ener_j, R2_j, _ = _read_e_theta_curve(fspn[1], fener[1], iat[1])
    _, ener_ij, R2_ij, r_ij  = _read_e_theta_curve(fspn[2], fener[2], iat, fcoord[0])
    print(f'Pearson correlation coefficients for perturbation are \
\ni: {R2_i}, \nj: {R2_j} and \ni and j: {R2_ij}, respectively.')
    if not np.allclose(theta_i, theta_j):
        raise ValueError('theta_i and theta_j are not the same')
    # thetaij should be the half of the theta_i or theta_j, so
    # they are not equal.

    if len(set(r_ij)) > 1:
        raise ValueError('inconsistency in the distance between the atoms')
    r_ij = r_ij[0]

    theta = np.abs(theta_i.flatten())
    ener = ener_ij - (ener_i + ener_j)
    ener = ener.flatten()

    return theta, ener, r_ij

def cal_Jij_r(fspn, fener, iat, fcoord = None, lin_thr = 1e-1):
    '''
    calculate the distance-dependent magnetic exchange coefficient Jij(r)
    '''
    theta, ener, dist = parse(fspn, fener, iat, fcoord)
    
    # calculate the Pearson correlation coefficient of the dEij ~ theta^2/2
    R2_r = np.corrcoef(theta**2/2, ener)[0, 1]
    if abs(abs(R2_r) - 1.0) > lin_thr:
        raise ValueError(f'low linearity {R2_r} between E ~ theta^2\
 of \natom-pair: {iat} \nspaced with distance: {dist:.2f} (Angstrom)')
    # linear fitting
    print(f'Perform linear fitting for Jij(r) at distance: {dist} (in Angstrom)')
    print(f'{"theta":>10} | {"energy":>10}')
    print('-'*23)
    for t, e in zip(theta, ener):
        print(f'{t:<10.4f} | {e:<10.4f}')

    Jij = np.polyfit(theta**2/2, ener, 1)[0]
    return Jij, R2_r, dist

def pertgen(proto, 
            fmt,
            pertkinds, 
            pertmags, 
            jobdir,
            out_fmt='deepmd/npy',
            overwrite=False,
            prefix='out'):
    '''generate the deltaspin related perturbated structures, return the
    dir in which one or more systems are generated. Structure:
    ```
    jobdir/
    `-- system1
        |-- set.000
        |   |-- box.npy
        |   |-- coord.npy
        |   `-- spin.npy
        |-- type.raw
        `-- type_map.raw
    `-- system2
        |-- set.000
        |   |-- box.npy
        |   |-- coord.npy
        |   `-- spin.npy
        |-- type.raw
        `-- type_map.raw
    `...
    ```
    
    Returns
    -------
    str : jobdir
        the directory in which the perturbated structures are generated
    '''
    _pert_kernel_impl(fn=proto,
                      fmt=fmt,
                      pertkinds=pertkinds,
                      pertmags=pertmags,
                      jobdir=jobdir,
                      out_fmt=out_fmt,
                      overwrite=overwrite,
                      fmodel=None,
                      prefix=prefix)
    return jobdir

def linear_response_calculation(proto,
                                fmt,
                                pertkinds,
                                pertmags,
                                overwrite,
                                calculator,
                                jobdir,
                                prefix,
                                **kwargs):
    '''perform series of calculations in the range where the energy has
    a linear response to the rotation of the magnetic moment.
    '''
    logging.info('Perform linear response calculation >>')
    info = 'Perform linear response calculation for the perturbated structures\n'
    info += f'prototype: {proto}\n'
    info += f'format: {fmt}\n'
    info += f'perturbated kinds: {pertkinds}\n'
    info += f'perturbated magnitudes: {pertmags}\n'
    info += f'overwrite: {overwrite}\n'
    info += f'calculator: {calculator}\n'
    info += f'jobdir: {jobdir}\n'
    info += f'prefix: {prefix}\n'
    logging.info(info)
    
    # first generate the perturbated structures
    jobdir = pertgen(proto=proto,
                     fmt=fmt,
                     pertkinds=pertkinds,
                     pertmags=pertmags,
                     jobdir=jobdir,
                     overwrite=overwrite,
                     prefix=prefix)
    
    # then perform calculation based on the generated structures
    if calculator.endswith('.pth'): # calculate with deepmd
        _predict_dp_impl(fmodel=calculator, testdir=jobdir, prefix=prefix)
    elif 'abacus' in calculator:
        _predict_dptest_abacus_impl(command=calculator,
                                    jobdir=jobdir,
                                    prefix=prefix,
                                    **kwargs)
    else:
        raise ValueError(f'unsupported calculator: {calculator}')
    
    logging.info('<< Linear response calculation done')
    return None

def _predict_dptest_abacus_impl(command, jobdir, prefix, **kwargs):
    '''counter part of deltaspin.predict function. Perform calculations
    based on each of structures defined in the `jobdir`. Because this is
    merely a substitute of deltaspin.predict, what in the `jobdir` are
    those systems stored in format `deepmd/npy`.
    
    This is a pure workflow function, unittest is not necessary.
    
    Parameters
    ----------
    command : str
        the command to run ABACUS. It is recommended to use the
        `mpirun -np [n] abacus | tee abacus.log` command, it can
        both run abacus, keep information printed on the screen and
        save the information to a file.
    jobdir : str
        the directory in which dpdata.MultiSystems are stored. Different
        systems will be saved in different folders, each folder will be
        named by the chemical composition (see the function deltaspin.
        _deepmd_signature for details).
    prefix : str
        the prefix of the output file. After all calculations, there will
        be files named as `[prefix].e_peratom.out`, `[prefix].out`, ...
        in cwd.
    kwargs : dict
        the keyword arguments, including:
        - fdft : str
            the ABACUS INPUT file, it should be a file in the `jobdir`
        - fpsp : list of str
            the list of pseudopotentials for elements. It should be a list
            of files in the `jobdir`
        - forb : list of str
            the list of orbitals for elements. It should be a list of files
            in the `jobdir`
    
    Returns
    -------
    '''
    logging.info('Predicting with ABACUS >>')
    
    # sanity check
    multisystems = [f for f in os.listdir(jobdir) if is_dpdata_dir(os.path.join(jobdir, f))]
    if len(multisystems) > 1:
        logging.warning(f'More than one structural proto: {len(multisystems)}')

    if len(multisystems) != 1:
        errmsg = f'Number of systems to predict is unexpected in {jobdir}: {len(multisystems)}'
        logging.error(errmsg)
        raise RuntimeError(errmsg)
            
    fdft = kwargs.get('fdft')
    if fdft is None:
        errmsg = 'fdft (the ABACUS INPUT) is not given'
        logging.error(errmsg)
        raise ValueError(errmsg)
    fpsp = kwargs.get('fpsp')
    if fpsp is None:
        errmsg = 'fpsp (list of pseudopotentials for elements) is not given'
        logging.error(errmsg)
        raise ValueError(errmsg)
    forb = kwargs.get('forb')
    if forb is None:
        errmsg = 'forb (list of orbitals for elements) is not given. \
It is not possible to perform deltaspin DFT calculation w/o it.'
        logging.error(errmsg)
        raise ValueError(errmsg)

    logging.info(f'ABACUS INPUT file: {fdft}\nPseudopotentials: {fpsp}\nOrbitals: {forb}')

    for idir, dpdir in enumerate(multisystems):
        e_peratom_predict, e_predict, fm_predict, fr_predict = [], [], [], []
        v_peratom_predict, v_predict = [], [] # ABACUS never outputs Virial
        logging.info(f'Predicting for {dpdir} >>')
        for idx, fstru in enumerate(to_abacus(os.path.join(jobdir, dpdir), 
                                              os.path.join(jobdir, dpdir + '-abacus'),
                                              spin_constrain=True,
                                              fpsp=[os.path.basename(f) for f in fpsp],
                                              forb=[os.path.basename(f) for f in forb])): # different structures...
            # perform the abacus calculation
            abacus_dir = os.path.join(os.path.dirname(fstru), dpdir+f'-{idx}')
            print(abacus_dir)
            if not isfinished(abacus_dir):
                os.makedirs(abacus_dir) # it is not okay to use the same name
                shutil.move(fstru, os.path.join(abacus_dir, 'STRU')) # move the stru file
                shutil.copy(fdft, abacus_dir)
                for f in fpsp:
                    shutil.copy(f, abacus_dir)
                for f in forb:
                    shutil.copy(f, abacus_dir)
                cwd = os.getcwd()
                os.chdir(abacus_dir)
                os.system(command)
                os.chdir(cwd)
            else:
                logging.info(f'Calculation defined in {abacus_dir} is already finished, skip it.')
            # save data
            e, e_peratom, fm, fr, v, v_peratom = _read_abacus(abacus_dir)
            e_predict.append(e)
            e_peratom_predict.append(e_peratom)
            fm_predict.append(fm)
            fr_predict.append(fr)
            v_predict.append(v)
            v_peratom_predict.append(v_peratom)
        logging.info(f'<< Predicted for {dpdir}')
    
        # write the results to files
        e = np.array(e_predict).reshape(-1, 1)
        e_peratom = np.array(e_peratom_predict).reshape(-1, 1)
        fm = np.array(fm_predict).reshape(-1, 3)
        fr = np.array(fr_predict).reshape(-1, 3)
        v = np.array(v_predict).reshape(-1, 9)
        v_peratom = np.array(v_peratom_predict).reshape(-1, 9)
        
        prfx = prefix if len(multisystems) == 1 else f'{prefix}-{idir}'
        f1 = write_dp_energies(prfx, e, cwd, None, False)
        f2 = write_dp_energies(prfx, e_peratom, cwd, None, True)
        f3 = write_dp_forces(prfx, fr, cwd, None, False)
        f4 = write_dp_forces(prfx, fm, cwd, None, True)
        f5 = write_dp_virials(prfx, v, cwd, None, False)
        f6 = write_dp_virials(prfx, v_peratom, cwd, None, True)
        logging.info(f'Predicted results are saved in files: \n\
{f1}\n{f2}\n{f3}\n{f4}\n{f5}\n{f6}')
        
    logging.info('<< Predicting done')
    return None

def _read_abacus(jobdir):
    '''a wrapper function to read e, e_peratom, fm, fr, v, v_peratom from
    the ABACUS log file.
    '''
    fdft = os.path.join(jobdir, 'INPUT')
    dft = read_fdft(fdft)
    suffix = dft.get('suffix', 'ABACUS')
    cal_type = dft.get('calculation', 'scf')
    if cal_type != 'scf':
        logging.warning(f'calculation type is not scf: {cal_type}')
    
    flog = os.path.join(jobdir, f'OUT.{suffix}', f'running_{cal_type}.log')
    # then read
    e = read_abacus_scf_energy(flog)
    nat = read_natoms_from_abacus_log(flog)
    fr = read_abacus_forces(flog)[-1][1] # only the last frame, without the element
    fm = np.zeros_like(fr)
    virial = np.zeros((3, 3))
    
    return e, e/nat, fm, fr, virial, virial/nat

def cal_magmom_exch_const(fspn, fener, iat, fcoord = None, lin_thr = 1e-1):
    '''
    calculate the magnetic exchange coefficient Jij(r) for a pair of atoms
    
    Parameters
    ----------
    fspn : list of list of str
        the file paths of the spin configurations. [r][i/j/ij] -> str
        r denotes the distance between the two atoms, i/j/ij denotes
        the perturbation of atom i, atom j, and both atom i and j.
    fener : list of list of str
        the file paths of the energy files. [r][i/j/ij] -> str
    iat : list of list of int
        the index of the atoms. [r][0/1] -> int
    
    Returns
    -------
    Jij : np.array
        the distance-dependent magnetic exchange coefficient Jij(r)
    '''
    # sanity check
    if len(fspn) != len(fener):
        raise ValueError('fspn and fener should have the same length')
    nr = len(fspn)
    if any([len(iat_) != 2 for iat_ in iat]):
        raise ValueError('iat should have length 2 for each element')
    if len(iat) != nr:
        raise ValueError('iat should have the same length as fspn')
    
    Jij = np.zeros(nr)
    for i in range(nr):
        fc = fcoord[i] if fcoord is not None else None
        Jij[i], r, dist = cal_Jij_r(fspn[i], fener[i], iat[i], fc, lin_thr)
        print(f'Jij({i}) = {Jij[i]*1e3:.4e} meV, \n\
R^2 = {r:.2f}, \n\
atomic distance = {dist:.2f} Angstrom\n')
    return Jij

def collect(root, iat, jat, prefix, nnn, signature):
    '''initialize the workflow by returning all the file paths
    
    Notes
    -----
    this function is hard-coded
    '''
    job = f'{signature}/set.000' # VERY DANGEROUS IMPLEMENTATION!!!
    fspn, fener = [], []
    fcoord = []
    for i in range(nnn):
        fspn.append([f'{root}/{prefix}-Ji-pt1/{job}/spin.npy', 
                     f'{root}/{prefix}-Jj-pt{i+1}/{job}/spin.npy', 
                     f'{root}/{prefix}-Jij-pt{i+1}/{job}/spin.npy'])
        fener.append([f'{root}/{prefix}-Ji-pt1.e_peratom.out',
                      f'{root}/{prefix}-Jj-pt{i+1}.e_peratom.out',
                      f'{root}/{prefix}-Jij-pt{i+1}.e_peratom.out'])
        fcoord.append([f'{root}/{prefix}-Jij-pt{i+1}/{job}/coord.npy'])
    iat = list(zip([iat]*len(jat), jat))
    Rij = [2.48, 2.86, 4.04, 4.75, 4.95]
    return fspn, fener, fcoord, iat, Rij

def Liechtenstein(proto, overwrite, fmt, calculator, prefix, 
                  iat, jat, 
                  mag = 2.4,
                  infinitesimal = 5, n = 7,
                  **kwargs):
    '''a hard-coded workflow to prepare the data shown in Fig.7(a) in
    Rinaldi M, Mrovec M, Bochkarev A, et al.
    Non-collinear magnetic atomic cluster expansion for iron[J].
    npj Computational Materials, 2024, 10(1): 12.

    Parameters
    ----------
    iat : int
        the index of atom at the Origin, always 0
    jat : list of int
        the index of atom at the Destination
    '''
    flog = f'DeltaspinPESGenerator@{time.strftime("%Y%m%d-%H%M%S")}.log'
    logging.basicConfig(filename=flog, 
                        level=logging.INFO, 
                        format='%(asctime)s - %(message)s')
    # astonishing! cp2kdata overwrited the logging???
    logging.info(f'log file: {flog}')

    # sanity check
    if not isinstance(iat, int):
        errmsg = f'iat must be an integer, but got {type(iat)}'
        logging.error(errmsg)
        raise ValueError(errmsg)

    cell = [0]
    
    # first calculate the one-atom perturbation
    for i, idx in enumerate([iat] + jat):
        pertkinds = ['cell', f'magmom:atom:{idx+1}']

        pref = 1 if i == 0 else -1
        mags = [np.array(sph_to_cart(mag, pref*a, 0)) for a \
                in np.linspace(0, np.deg2rad(infinitesimal), n, endpoint=True)]
        pertmags = [cell, mags]

        jobdir = f'{prefix}-J'
        jobdir += 'i-pt1' if i == iat else f'j-pt{i}'
        logging.info(f'generated jobs will be in folder: {jobdir}')

        linear_response_calculation(proto, fmt, pertkinds, pertmags, overwrite, calculator,
                                    jobdir=jobdir, prefix=jobdir, **kwargs) 
        # for abacus, the kwargs are necessary

    fcoord = os.path.join(f'{prefix}-Ji-pt1', f'{_deepmd_signature(proto)}/set.000/coord.npy')
    coord = np.load(fcoord)[0].reshape(-1, 3) # only the first frame

    # then calculate the two-atom perturbation. This is a little bit
    # complicated, since we only want to calculate the distance-dependent
    # exchange coefficient Jij(r) for a pair of atoms, so the spins of
    # the atoms at the Origin and the Destination should be vertical
    # to the normal vector connecting the two atoms before rotation.
    #
    # Therefore, first we need to rotate all spins of atoms to the 
    # direction that vertical to the normal vector connecting the
    # Origin and the Destination, then rotate the spins of the atoms
    # at the Origin and the Destination to the opposite direction
    # with the same angle.
    for i, idx in enumerate(jat):
        pertkinds = ['cell', f'magmom:atom:{iat+1},{idx+1}']

        # normal vector connecting iat and jat
        eRij = coord[idx] - coord[iat]
        eRij /= np.linalg.norm(eRij)
        logging.info(f'normal vector connecting iat and jat: {eRij}')

        # normal vector of the rotated spin, should be perpendicular
        # to the normal vector connecting iat and jat (eRij)
        # thus for simplicity, we rotate the vector eRij by pi/2
        # as the normal vector to the rotated spin
        _, thetaRij, phiRij = cart_to_sph(*(eRij.tolist()))
        emag = np.array(sph_to_cart(1, thetaRij + np.pi/2, phiRij))
        logging.info(f'normal vector to the rotated spin: {emag}')
        
        if np.dot(emag, eRij) > 1e-6:
            errmsg = f'normal vector to the rotated spin is not perpendicular to the normal vector connecting iat and jat'
            logging.error(errmsg)
            raise ValueError(errmsg)

        # rotated magnetic moment
        logging.info(f'overwrite refresh to: {overwrite*emag}, \nits norm: {np.linalg.norm(overwrite*emag)}')
        mags = [np.array([
            (emag * np.cos(a) + eRij * np.sin(-a)) * mag,
            (emag * np.cos(a) + eRij * np.sin(+a)) * mag,
        ]) for a in np.linspace(0, np.deg2rad(infinitesimal/2), n, endpoint=True)]
        
        pertmags = [cell, mags]

        jobdir = f'{prefix}-Jij-pt{i+1}'
        logging.info(f'generated jobs will be in folder: {jobdir}')
        
        logging.info(f'Magmom rotated w.r.t. `overwrite` vector with angles:')
        for m in mags:
            dot1, dot2 = np.dot(m[0, :], emag)/mag, np.dot(m[1, :], emag)/mag
            a1 = 0. if np.abs(dot1 - 1) < 1e-6 else -np.rad2deg(np.arccos(dot1))
            a2 = 0. if np.abs(dot2 - 1) < 1e-6 else np.rad2deg(np.arccos(dot2))
            logging.info(f'{a1:>6.2f} {a2:>6.2f}')

        linear_response_calculation(proto, fmt, pertkinds, pertmags, overwrite*emag, calculator,
                                    jobdir=jobdir, prefix=jobdir, **kwargs)
    
    logging.info('done')
    logging.shutdown()
    return None

def build_cell(elem, abc, atoms, nnn, outdir=None):
    '''
    build a supercell for the calculation of magnetic exchange
    coefficient Jij(r).
    
    Parameters
    ----------
    elem : str
        the element of the atoms
    abc : list
        the lattice constant of the supercell
    atoms : list
        the positions of the atoms in the unit cell, in fractional
        coordinates
    nnn : int
        the number of shells of n.n. (the nearest-neighbors)
        to calculate the magnetic exchange coefficient Jij(r)
    '''
    outdir = outdir if outdir is not None else os.getcwd()
    
    # calculate the number of atoms in the supercell
    n = supercell_count(abc, nnn, atoms) + 1
    # then after plus one, it is the time that we duplicate the
    # original cell
    
    # build a cell object
    cell = build_abacus_supercell(elem, abc, atoms, n)
    nat = n**3 * len(atoms)
    
    # write files...
    fn = os.path.join(outdir, f'{elem}-{"".join(map(str, [n]*3))}')
    write_abacus_stru(cell, fn)
    
    # return the file name
    return fn

def main(elem, abc, atoms, nnn, mag, overwrite, calculator, infinitesimal, n, **kwargs):
    '''the main workflow'''
    proto = build_cell(elem, abc, atoms, nnn)
    iat = 0
    stru = read_stru(proto)
    jat = make_shell_spikes(stru, nnn, iat, elem, True)
    
    _ = Liechtenstein(proto=proto,
                      overwrite=overwrite,
                      fmt='abacus/stru',
                      calculator=calculator,
                      prefix='Fig7a',
                      iat=iat,
                      jat=jat,
                      mag=mag,
                      infinitesimal=infinitesimal, 
                      n=n,
                      **kwargs)
    
    # calculate the distance between atom-pairs
    ielem = [typ[0] for typ in stru['ATOMIC_SPECIES']].index(elem)
    alat = float(stru['LATTICE_CONSTANT'][0][0]) # sorry...
    coordj = np.array([stru['ATOMIC_POSITIONS']['atoms'][ielem]['xyz'][i] for i in jat])
    cell = np.array(stru['LATTICE_VECTORS']).astype(float) * alat # in Bohr
    factor = convert_length_unit(1.0, 'bohr', 'angstrom')
    coordj = coordj @ cell * factor
    origin = np.array(stru['ATOMIC_POSITIONS']['atoms'][ielem]['xyz'][iat]) @ cell * factor
    Rij = np.linalg.norm(coordj - origin, axis=1)
    
    # postprocess.
    fspn, fener, fcoord, ijat, _ = collect('.', iat, jat, 'Fig7a', nnn, _deepmd_signature(proto))
    Jij = cal_magmom_exch_const(fspn, fener, ijat, fcoord)
    
    return Rij, Jij

class MagmomExchConstTest(unittest.TestCase):
    
    def test_build_cell(self):
        from abacus import read_stru
        # a ground truth is, if we want to simulate the
        # 5-th shell of atoms, for BCC Fe, we at least
        # need a 3x3x3 supercell
        elem = 'Fe'
        abc = [2.87, 2.87, 2.87]
        atoms = [[0, 0, 0], [0.5, 0.5, 0.5]]
        nnn = 5
        fn = build_cell(elem, abc, atoms, nnn)
        self.assertTrue(os.path.exists(fn))
        stru = read_stru(fn)
        os.remove(fn)
        self.assertEqual(len(stru['ATOMIC_POSITIONS']['atoms'][0]['xyz']), 3**3*2)
        lat = np.array(stru['LATTICE_VECTORS']).astype(float).tolist()
        self.assertEqual(lat, np.diag([2.87*3]*3).tolist())

    def test_make_shell_spikes(self):
        from abacus import read_stru
        from structure import _radial_count as radial_count
        # a ground truth is, if we want to simulate the
        # 5-th shell of atoms, for BCC Fe, we at least
        # need a 3x3x3 supercell
        elem = 'Fe'
        abc = [2.87, 2.87, 2.87]
        atoms = [[0, 0, 0], [0.5, 0.5, 0.5]]
        nnn = 5
        fn = build_cell(elem, abc, atoms, nnn)
        self.assertTrue(os.path.exists(fn))
        stru = read_stru(fn)
        os.remove(fn)
        
        # make the shell spikes
        spikes = make_shell_spikes(stru, nnn, 0, 'Fe', True)
        dist, _, _ = radial_count(abc, nnn, atoms)
        pos = np.array(stru['ATOMIC_POSITIONS']['atoms'][0]['xyz'])
        origin = pos[0]
        disp = np.array([p - origin for p in pos[spikes]]) @ (np.diag(abc) * 3)
        self.assertTrue(np.allclose(dist, np.linalg.norm(disp, axis=1)))

    def test_pertgen(self):
        proto = 'phonopy-BCC-Fe/BCC-Fe-primitive'
        jobdir = pertgen(proto=proto,
                         fmt='abacus/stru',
                         pertkinds=['cell', 'magmom:atom:1'],
                         pertmags=[[0], np.linspace(0, 2.4, 7).tolist()],
                         jobdir='magmom-exch-const-test',
                         overwrite=2.4,
                         prefix='test')
        self.assertTrue(os.path.exists(jobdir))
        shutil.rmtree(jobdir)

if __name__ == '__main__':

    test = init()
    unittest.main(exit=test)
    
    Rij, Jij = main(
        # basic structure definition
        # --------------------------
        elem='Fe',       # the element symbol, now only unary system is supported :(
        abc=[2.86304] * 3,
        # the lattice, can be a list of 3 numbers (indicating a cubic cell)
        # or a np.array of shape (3, 3) (indicating the lattice vectors)
        atoms=[[0.0000, 0.0000, 0.0000], 
               [0.5000, 0.5000, 0.5000]],
        # direct coordinates of the atoms in the cell

        # heisenberg model
        # ----------------
        nnn=1,           # the number of shells of n.n. (the nearest-neighbors) to consider
        mag=2.4,         # the magmom of the atom(s) whose magmom to be rotated
        infinitesimal=5, # the maximal angle of the rotation
        n=5,             # the number of steps in the rotation
                    
        # magmom of other atoms
        # ---------------------
        overwrite=2.4,
        
        # energy calculator
        # -----------------
        calculator='mpirun -np 16 abacus',
        # note: this can be set either to the path to a dp model, or to a command that run
        # ABACUS, in the latter case, you need to provide additional parameters in the following
        
        # DFT additional parameters
        # -------------------------
        fdft='heisenberg-Fe/INPUT',
        fpsp=['pp_orb/Fe.upf'],
        forb=['pp_orb/Fe_gga_8au_200.0Ry_4s2p2d1f.orb']
        )

    print(f'Jij = {Jij*1e3} meV')

    # then plot the Jij(r) curve
    fontsize = 14
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(Rij, Jij*1e3, 
            marker='o', 
            linestyle='-',
            linewidth=2,
            color='red', 
            markersize=8)
    ax.set_xlabel('$R_{0j}$ ($\\AA$)', fontsize=fontsize)
    ax.set_ylabel('$J_{ij}$ (meV)', fontsize=fontsize)
    plt.tight_layout()
    plt.savefig('fig7a.png', dpi=300)
    plt.close()
