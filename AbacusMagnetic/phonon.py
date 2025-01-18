'''
a light-weight phonon spectrum calculator

Warning
-------
ABACUS force calculator is not supported yet

Version
-------
2024/12/20 19:03

Prerequisites
-------------
All external requirements will be listed here
dpdata
pymatgen
phonopy
'''

# external packages
try: # phonopy is not used as a python module
    from phonopy import Phonopy
except ImportError:
    raise ImportError("phonopy is required to run this module")
import dpdata
import numpy as np

# internal packages
import re
import os
import shutil
import unittest
import uuid
import logging
import time
from enum import Enum

# home-made packages
from deltaspin import main as dp_kernel
from deltaspin import _deepmd_signature
from deltaspin import read_forces as read_dp_forces
from abacus import read_forces as read_abacus_forces
from abacus import write_forces as write_abacus_forces
from abacus import read_fdft
from utils import init

def build_phonopy_supercell(fn, dim, workdir=None, outdir=None):
    '''
    build phonopy supercell from ABACUS STRU file
    
    Parameters
    ----------
    fn : str
        ABACUS STRU file name
    dim : list
        supercell dimension
        
    Returns
    -------
    list
        list of phonopy generated perturbed structures. Note: calculate
        forces of atoms in these structures and do not relax them.
    '''
    logging.info(f'Build phonopy supercell from {fn} with dimension {dim} >>')
    def phononpy_files(path):
        '''in given folder, scan all phonopy files'''
        pat = r'^STRU\-\d\d\d$'
        return [i for i in os.listdir(path) if (re.match(pat, i) or i == 'phonopy_disp.yaml')]
    
    # workdir
    workdir = workdir or os.getcwd()
    # outdir
    outdir = outdir or os.getcwd()
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    
    cwd = os.getcwd()
    os.chdir(workdir)
    files = phononpy_files(workdir)
    if len(files) == 0:
        if not (isinstance(dim, list) and len(dim) == 3 and all([isinstance(i, int) for i in dim])):
            errmsg = "dim should be a list of 3 integers"
            logging.error(errmsg)
            raise ValueError(errmsg)
        
        cmd = f'phonopy --abacus -c {fn} -d --dim=\"{" ".join([str(i) for i in dim])}\"'
        logging.info(f'Run phonopy command: `{cmd}`')
        os.system(cmd)
        files = phononpy_files(workdir)
    else:
        logging.warning(f'There are already phonopy files in the {outdir}')
    files = [f for f in files if re.match(r'^STRU\-\d\d\d$', f)]
    
    for f in files:
        os.rename(f, os.path.join(outdir, f))
    os.chdir(cwd)

    logging.info(f'<< Build phonopy supercell from {fn} with dimension {dim}')
    return files

def dp_calculator(calculator, fn, **kwargs):
    '''calculate the forces of structures perturbated by pertkinds and pertmags
    on present structure
    '''
    # run the force calculator
    prefix = os.path.basename(fn)
    jobdir = f'phonon-{prefix}'
    dp_kernel(fn, # the prototype structure in ABACUS STRU format
              'abacus/stru', 
              pertkinds=kwargs.get('pertkinds', ['cell']), 
              pertmags=kwargs.get('pertmags', [[0]]),
              jobdir=jobdir, # where the structure acceptable by deepmd is stored
              out_fmt='deepmd/npy', 
              overwrite=kwargs.get('overwrite', False),
              fmodel=calculator, 
              prefix=prefix)

    # read general information
    stru_id = _deepmd_signature(fn)
    type_map = np.array(np.loadtxt(os.path.join(jobdir, stru_id, 'type_map.raw'), 
                                   dtype=str)).flatten().tolist()
    types = np.loadtxt(os.path.join(jobdir, stru_id, 'type.raw'), dtype=int).tolist()
    nat = len(types)
    elem = [type_map[i] for i in types]
    
    # read the forces
    workdir = os.path.dirname(fn)
    workdir = os.getcwd() if workdir == '' else workdir # that is where the output files are
    
    forces = np.loadtxt(os.path.join(workdir, f'{prefix}.fr.out'), 
                        skiprows=1)[:, -3:].astype(float).reshape(-1, nat, 3)
    mag_forces = np.loadtxt(os.path.join(workdir, f'{prefix}.fm.out'), 
                            skiprows=1)[:, -3:].astype(float).reshape(-1, nat, 3)
    # both forces and mag_forces are indexed with [pert][atom][x/y/z] -> float

    # move dumped files to the jobdir (good idea!)
    files = [i for i in os.listdir(workdir) if i.startswith(prefix) and i.endswith('.out')]
    for f in files:
        shutil.move(os.path.join(workdir, f), os.path.join(jobdir, f))
    
    return [(elem, i) for i in forces] #, [(elem, i) for i in mag_forces]

def _abacus_shortcut(workdir, unit = 'GPa'):
    '''a short cut, if the ABACUS run has been done before,
    the calculation will be skipped'''
    fn = os.path.join(workdir, 'OUT.ABACUS', 'running_scf.log')
    if os.path.exists(fn):
        return read_abacus_forces(fn)
    else:
        return None

def abacus_calculator(calculator, fn ,**kwargs):
    '''calculate the forces of present structure'''
    fdft = kwargs.get('fdft')
    fpsp = kwargs.get('fpsp')
    forb = kwargs.get('forb')
    if any([f is None for f in [fdft, fpsp, forb]]):
        raise ValueError('fdft (INPUT), fpsp (list of pseudopotentials), forb (list of orbitals) must be provided')
    fpsp = [fpsp] if isinstance(fpsp, str) else fpsp
    forb = [forb] if isinstance(forb, str) else forb

    # make the workdir, then perform calculation in the workdir
    workdir = 'dft-' + os.path.basename(fn)
    shortcut = _abacus_shortcut(workdir)
    if shortcut is not None:
        print(f'Found previous calculation in {workdir}, skipping')
        return shortcut
    else:
        print(f'Calculating stress with ABACUS in {workdir}')
    # it is not okay if there is already a folder with the same name
    if os.path.exists(workdir) and os.path.isdir(workdir):
        raise FileExistsError(f'The folder {workdir} already exists')
    os.makedirs(workdir, exist_ok=True)
    
    # copy files into the workdir
    shutil.copy(fn, os.path.join(workdir, 'STRU'))        
    shutil.copy(fdft, os.path.join(workdir, 'INPUT')) # may raise FileNotFoundError
    for f in fpsp:
        shutil.copy(f, workdir) # may raise FileNotFoundError
    for f in forb:
        shutil.copy(f, workdir) # may raise FileNotFoundError
        
    # start ABACUS
    cwd = os.getcwd()
    os.chdir(workdir)
    os.system(f'{calculator}')
    os.chdir(cwd)
    
    # get the suffix and calculation, so that determine the file name and outdir
    dft = read_fdft(fdft)
    suffix = dft.get('suffix', 'ABACUS')
    if dft.get('calculation', 'scf') != 'scf':
        raise ValueError('You should only perform SCF calculation rather than relaxing any atoms')
    fout = os.path.join(workdir, f'OUT.{suffix}', f'running_scf.log')
    return read_abacus_forces(fout)

def cal_force(calculator, fn, **kwargs):
    '''calculate the atomic forces of the given structure'''
    logging.info(f'Calculating atomic forces with {calculator} >>')
    
    out = None
    if calculator.endswith('.pth'): # Deltaspin model
        out = dp_calculator(calculator, fn, **kwargs)
    elif 'abacus' in calculator: # use ABACUS DFT
        out = abacus_calculator(calculator, fn, **kwargs)
    else:
        raise ValueError(f'Unsupported calculator: {calculator}')

    logging.info(f'<< Calculated atomic forces with {calculator}')
    return out

def main(relaxed, calculator, phonopy_dim, **kwargs):
    '''
    main workflow function
    
    Parameters
    ----------
    relaxed : str
        relaxed structure file name. Please always use the relaxed
        structure, otherwise there will be imaginary frequencies
    pertkinds : list[str]
        perturbation kinds, see annotation of `dp_calculator`
    pertmags : list[Iterable[float]]
        perturbation magnitudes, see annotation of `dp_calculator`
    fmodel : str
        the model file, used to predict the forces
    phonopy_dim : list
        phonopy supercell dimension, should be strictly a list of 3 integers
    overwrite : bool
        overwrite magmom of rest of atoms, default is False

    Roadmap
    -------
    this function will 
    1. prepare the perturbed structures that is needed by phononpy 
       to do supercell calculations, 
    2. calculate the forces for each perturbed structure,
    3. read the forces calculated and convert them to the format accepted by phonopy, 
    4. calculate the phonon spectrum
    '''
    # first, build the phonopy supercell
    files = build_phonopy_supercell(relaxed, phonopy_dim)
    # then, calculate the forces
    temp = []
    # DIM1: different phonon displacements
    for f in files: 
        force = cal_force(calculator, f, **kwargs)
        print(f'Calculated forces for all structures perturbated based on: {f}')
        # then, read the forces and convert them to the format that is acceptable by phonopy
        
        # DIM2: different perturbations
        for i, fr in enumerate(force): # i indexes the perturbations
            # write the forces to the cp2k format because it is clean
            fnfr = f'{f}-fr-1_{i}.log'
            write_abacus_forces(fnfr, *fr)
            temp.append(fnfr)
    
    fn1 = temp
    fn1 = np.array(fn1).reshape(len(files), -1).T.tolist()
    
    return fn1
    # then, post-process the forces frame-by-frame
    for f in forces:
        pass

# a enum class to constraint the option of the phonon calculation, named `Property`
class Property(Enum):
    '''the property of the phonon calculation
    
    Attributes
    ----------
    BAND : int
        calculate the band structure
    DOS : int
        calculate the density of states
    PDOS : int
        calculate the partial density of states
    TERMODYNAMICS : int
        calculate the thermodynamics properties, including Helmholtz free energy,
        head capacity, entropy, etc.
    '''
    BAND = 1
    DOS = 2
    PDOS = 3
    TERMODYNAMICS = 4

def postprocess(fn, properties: list[Property], outdir, prefix = 'DeltaSpinTest', **kwargs):
    '''do postprocess (various of phonon calculations)
    with phonopy
    
    Parameters
    ----------
    fn : str
        the file name of the forces
    properties : list[Property]
        the properties to calculate. It should be a list of `Property` enum
    prefix : str
        the prefix of the output files
    
    Returns
    -------
    list
        the output files
    '''
    logging.info(f'Postprocess the forces in {fn} >>')
    # read the forces from the file, output FORCE_SETS
    cmd = 'phonopy -f ' + ' '.join(fn)
    os.system(cmd)
    # then, calculate the phonon properties
    return
    out = []
    if Property.BAND in properties:
        logging.info('Calculate the band structure >>')
        qpath = kwargs.get('qpath')
        if qpath is None:
            logging.error('The qpath should be provided for the band calculation')
        else: # calculate the band structure
            qpath = np.array(qpath).flatten().tolist()
            cmd = f'phonopy-load --band \"' + ' '.join([str(q) for q in qpath]) + '\" -p -s'
            fbdyaml = os.path.join(outdir, f'{prefix}.band.yaml')
            fbdpdf = os.path.join(outdir, f'{prefix}.band.pdf')
            shutil.move('band.yaml', fbdyaml)
            shutil.move('band.pdf', fbdpdf)
            out.extend([fbdyaml, fbdpdf])
            logging.info(f'<< Calculate the band structure')
    
    if Property.DOS in properties:
        logging.info('Calculate the DOS >>')
        mesh = kwargs.get('mesh')
        if mesh is None:
            logging.error('The mesh should be provided for the DOS calculation')
        else: # calculate the DOS
            cmd = f'phonopy-load --mesh ' + ' '.join(mesh) + ' -s'
            fdosdat = os.path.join(outdir, f'{prefix}.total_dos.dat')
            shutil.move('total_dos.dat', fdosdat)
            out.append(fdosdat)
            logging.info(f'<< Calculate the DOS')
    
    if Property.PDOS in properties:
        logging.info('Calculate the PDOS >>')
        mesh = kwargs.get('mesh')
        if mesh is None:
            logging.error('The mesh should be provided for the PDOS calculation')
        else: # calculate the PDOS
            cmd = f'phonopy-load --mesh ' + ' '.join(mesh) + ' -p'
            fpdosyaml = os.path.join(outdir, f'{prefix}.mesh.yaml')
            shutil.move('mesh.yaml', fpdosyaml)
            out.append(fpdosyaml)
            logging.info(f'<< Calculate the PDOS')
        
    if Property.TERMODYNAMICS in properties:
        logging.info('Calculate the thermodynamics properties >>')
        mesh = kwargs.get('mesh')
        if mesh is None:
            logging.error('The mesh should be provided for the thermodynamics calculation')
        else: # calculate the thermodynamics properties
            cmd = f'phonopy-load --mesh ' + ' '.join(mesh)
            fthermo = os.path.join(outdir, f'{prefix}.thermal_properties.yaml')
            shutil.move('thermal_properties.yaml', fthermo)
            out.append(fthermo)
            logging.info(f'<< Calculate the thermodynamics properties')
    
    logging.info(f'<< Postprocess the forces in {fn}')

def cal_force_sets(fn):
    '''only postprocessing the forces by phonopy -f command,
    yields the FORCE_SETS file'''
    logging.info(f'Postprocess the forces in {fn} >>')
    fn = [fn] if isinstance(fn, str) else fn
    cmd = 'phonopy -f ' + ' '.join(fn)
    os.system(cmd)
    logging.info(f'<< Postprocess the forces in {fn}, output FORCE_SETS file.')

class TestPhonon(unittest.TestCase):
    
    def init_mytest(self):
        '''
        generate one ABACUS Structure
        '''
        s = '''ATOMIC_SPECIES
Fe 55.845 Fe.pbe-spn-rrkjus_psl.1.0.0.UPF

NUMERICAL_ORBITAL
Fe_gga_10au_100Ry_4s2p2d1f.orb

LATTICE_CONSTANT
1.889726125457828

LATTICE_VECTORS
3.405 0.000 0.000
0.000 3.405 0.000
0.000 0.000 3.405

ATOMIC_POSITIONS
Direct
Fe
0.00
2
0.00 0.00 0.00 m 1 1 1 mag 0 0 2.40 sc 1 1 1
0.50 0.50 0.50 m 1 1 1 mag 0 0 2.40 sc 1 1 1
'''
        fn = str(uuid.uuid4()) + '.stru'
        with open(fn, 'w') as f:
            f.write(s)
        return fn

    @unittest.skip('Not preferred to bother phonopy in unittest')
    def test_build_phononpy_supercell(self):
        '''
        test build_phonopy_supercell
        '''
        outdir = 'phonopy-test'
        code = str(uuid.uuid4())
        fn = self.init_mytest()
        files = build_phonopy_supercell(fn, [2, 2, 2], outdir=outdir)
        os.remove(fn)
        self.assertTrue(len(files) > 0)
        self.assertTrue(all([os.path.exists(os.path.join(outdir, i)) for i in files]))
        shutil.rmtree(outdir)

if __name__ == "__main__":
    
    flog = f'phononpy-{time.strftime("%Y%m%d-%H%M%S")}.log'
    logging.basicConfig(filename=flog, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    # will run this unittest each time to ensure the Phononpy module is working
    test = init()
    unittest.main(exit=test)
    
    out = main(
        # basic structure definition
        # --------------------------
        relaxed='phonopy-Diamond-Si/Diamond-Si-primitive-lcao',
        # the structure provided must be the fully relaxed one

        # Phonopy supercell method
        # ------------------------
        phonopy_dim=[4, 4, 4],
        
        # energy calculator
        # -----------------
        calculator='mpirun -np 16 abacus',

        # DFT additional parameters
        # -------------------------
        fdft='phonopy-Diamond-Si/INPUT',
        fpsp='phonopy-Diamond-Si/Si_ONCV_PBE-1.0.upf',
        forb='phonopy-Diamond-Si/Si_gga_10au_100Ry_2s2p1d.orb'
        )
    
    # all magnetic perturbation share the same phonopy_disp.yaml
    # but:
    # will magnetic moment breaks the symmetry, so that the phonon actually
    # has different dispersion?
    
    # Note: the BCC q-points should be
    # N-Gamma-H-P-Gamma
    # 0 0.5 0  0 0 0  -0.5 0.5 0.5  0.25 0.25 0.25  0 0 0
    # Run phonopy with
    # phonopy-load --band "0 0.5 0  0 0 0  -0.5 0.5 0.5  0.25 0.25 0.25  0 0 0" -p -s
        
    # for ipert, o in enumerate(out):
    #     outdir = f'phonon-{ipert}@{time.strftime("%Y%m%d-%H%M%S")}'
    #     logging.info(f'Postprocess the forces in magnetic perturbation {ipert} >>')
    #     fn = postprocess(o, [Property.BAND], outdir, prefix=f'phonon-{ipert}',
    #                      qpath=[[0, 0, 0], [0.5, 0.5, 0.5]],
    #                      mesh=[40, 40, 40])
    #     logging.info(f'Dumped files: {fn}')
    #     logging.info(f'<< Postprocess the forces in magnetic perturbation {ipert}')
    
    logging.shutdown()
    print(f'Log file is saved as {flog}')
    