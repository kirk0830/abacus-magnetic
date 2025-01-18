'''
a light-weight elastic coefficient calculation driver

Version
-------
2024/12/20 19:03

Prerequisites
-------------
All external requirements will be listed here
dpdata
pymatgen

What's done
-----------
Structure deformation, export and format conversion.
Now it is capable to 
-> read the structure from ABACUS STRU file,
-> perform deformation,
-> get the deformation information and the deformed structure as dpdata.System object.
Functions:
1. read_structure
   read ABACUS STRU file as the structure to test the elastic properties
   and convert it to pymatgen structure object
2. pymatgen2dpdata
   convert the pymatgen structure object to dpdata.System object, support
   the normal structure and the deformed structure
3. deform
   deform the structure with given strain, and return the deformed structure
4. write_pymatgen_deform
   write the deformed structure to ABACUS STRU file, with the deform
   information as the annotation.
5. read_deform_as_dpdata
   read the deformed structure from ABACUS STRU file, and return the
   dpdata.System object with the deformation information

ABACUS as stress calculator
Now it is capable to calculate stress via ABACUS-DFT
Functions:
1. abacus_calculator
   calculate the stress with ABACUS-DFT, with the file name of the structure
   file, the ABACUS INPUT file, the pseudopotential file, and the orbital file

Elastic properties calculation
Now it is capable to
-> according to calculated stress and strain, calculate the elastic properties
   as dict
Functions:
1. calculate
   calculate the elastic properties with given strain and stress


TBD
---
up to 2024/12/18, the calculation of stress is not impelmented yet
for DeltaSpin model, therefore, the calculation of elastic properties
are not integrated and automated yet.
'''
# external packages
import dpdata
from pymatgen.analysis.elasticity.strain import DeformedStructureSet, Strain
from pymatgen.analysis.elasticity.stress import Stress
from pymatgen.analysis.elasticity.elastic import ElasticTensor
from pymatgen.core.structure import Structure
from pymatgen.core.tensors import Tensor
from pymatgen.core.operations import SymmOp
import numpy as np

# in-build python packages
import uuid
import os
import unittest
import shutil
import re
import logging
import time

# home-made packages
from deltaspin import main as dp_kernel
from abacus import read_fdft, _kmeshgen, _write_abacus_kpt
from abacus import read_stress as read_abacus_stress
from utils import convert_pressure_unit as _convert_pressure_unit
from utils import _dict_as_text, _text_as_dict, init

def _mysed_(fn, pattern, replace):
    '''execute the sed command'''
    os.system(f'sed -i "s/{pattern}/{replace}/g" {fn}')

def read_structure(fn, fmt = 'abacus/stru'):
    '''
    Read structure from file. For the scope that the `fmt` supports,
    please refer to the documentation of `dpdata.System`.

    Parameters
    ----------
    fn : str
        The file name of the structure file
    fmt : str
        The format of the structure file
    
    Returns
    -------
    Structure
        The structure object, see `pymatgen.core.structure.Structure`
    '''
    logging.info(f'Reading structure file: {fn} >>')
    temp = dpdata.System(fn, fmt = fmt)
    code = str(uuid.uuid4()).replace('-', '')
    fn = code[:len(code)//2] + '-POSCAR-' + code[len(code)//2:]
    temp.to('vasp/poscar', fn)
    temp = Structure.from_file(fn)
    os.remove(fn)
    logging.info(f'<< Read structure file: {fn}')
    return temp

def _ieee_standardize(obj: Structure):
    '''
    overwrite the structure file with IEEE standardization
    '''
    logging.info(f'Standardizing the structure with IEEE standard >>')
    out = obj.copy()
    rot = Tensor.get_ieee_rotation(out)
    op = SymmOp.from_rotation_and_translation(rot)
    out.apply_operation(op)
    logging.info(f'<< Standardized the structure with IEEE standard')
    return out

def _make_primitive(obj: Structure, tol = 0.5):
    '''make the cell to be primitive after the search of primitive cell'''
    logging.info(f'Making the structure to be primitive >>')
    out = obj.get_primitive_structure(tolerance=tol)
    logging.info(f'<< Made the structure to be primitive')
    return out

def pymatgen2dpdata(obj: Structure|DeformedStructureSet):
    '''
    Convert the structure object from pymatgen, to
    the format that dpdata supports

    Parameters
    ----------
    obj : Structure|DeformedStructureSet
        The structure object, see `pymatgen.core.structure.Structure`
        or see the return of `read_structure`

    Returns
    -------
    list of tuple of dpdata.System and dict
        The dpdata.System object and the deformation information
    '''
    logging.info(f'Converting pymatgen structure to dpdata.System >>')
    out = []
    if isinstance(obj, Structure):
        logging.info(f'Converting normal structure, no deformation information')
        # normal structure, without any additional information
        code = str(uuid.uuid4()).replace('-', '')
        fn = code[:len(code)//2] + '-POSCAR-' + code[len(code)//2:]
        obj.to(fn)
        out = [(dpdata.System(fn, fmt = 'vasp/poscar'), {})]
        os.remove(fn)
    elif isinstance(obj, DeformedStructureSet):
        # deformed structure, with additional deformation information
        # will be stored in additional description-{i}.json files
        n = len(obj)
        logging.info(f'Converting deformed structure ({n}), with deformation information')
        for i in range(n):
            code = str(uuid.uuid4()).replace('-', '')
            fn = code[:len(code)//2] + '-POSCAR-' + code[len(code)//2:]
            obj[i].to(fn)
            out.append((dpdata.System(fn, fmt = 'vasp/poscar'), 
                        obj.deformations[i].as_dict(voigt=True)))
            os.remove(fn)
    else:
        errmsg = f'Invalid type: {type(obj)}'
        logging.error(errmsg)
        raise TypeError(errmsg)

    logging.info(f'<< Converted pymatgen structure to dpdata.System')
    return out

def ieee_structure(fn, fmt = 'abacus/stru', primitive = True, tol = 0.5):
    '''re-generate the structure file with IEEE standardization
    for elastic properties calculation
    
    Parameters
    ----------
    fn : str
        The file name of the structure file
    fmt : str
        The format of the structure file
    primitive : bool
        Whether to make the cell to be primitive or not
    tol : float
        The tolerance for making the cell to be primitive
    
    Returns
    -------
    str
        The file name of the standardized structure file
    '''
    logging.info(f'Standardizing the structure file: {fn} >>')
    temp = read_structure(fn, fmt)           # pymatgen.core.structure.Structure instance
    if primitive:
        temp = _make_primitive(temp, tol)    # make the cell to be primitive
    temp = _ieee_standardize(temp)           # standardize the structure
    temp = pymatgen2dpdata(temp)[0][0]       # dpdata.System instance
    fn = os.path.join(os.path.dirname(fn), 'IEEE-' + os.path.basename(fn))
    temp.to(fmt, fn)
    logging.info(f'<< Standardized structure file: {fn}')
    return fn

def deform(obj: Structure, norm = 1e-2, shear = 1e-2):
    '''
    Deform the structure with given strain.
    This function will always return 24 deformed structures.

    Parameters
    ----------
    obj : Structure
        The structure object, see `pymatgen.core.structure.Structure`
        or see the return of `read_structure`
    norm : float
        The norm strain, default is 1e-2
    shear : float
        The shear strain, default is 1e-2
    
    Returns
    -------
    DeformedStructureSet
        The deformed structure object, see 
        `pymatgen.analysis.elasticity.strain.DeformedStructureSet`
    '''
    logging.info(f'Deforming structure with norm = {norm}, shear = {shear} >>')
    norm = norm if isinstance(norm, list) else \
    [     -norm, -0.5 * norm,
     0.5 * norm,        norm]
    shear = shear if isinstance(shear, list) else \
    [     -shear, -0.5 * shear,
     0.5 * shear,        shear]

    # temp = obj.copy()
    temp = DeformedStructureSet(obj, 
                                symmetry=False,
                                norm_strains=norm,
                                shear_strains=shear)
    logging.info(f'Number of deformed structures: {len(temp)}')
    logging.info(f'<< Deformed structure with norm = {norm}, shear = {shear}')
    return temp

def _annotate_abacus_stru(fn, annotation: dict):
    '''
    Annotate the structure file with additional information

    Parameters
    ----------
    fn : str
        The file name of the structure file
    annotation : dict
        The annotation information
    '''
    with open(fn, 'r') as f:
        data = f.readlines()
    anno_text = _dict_as_text(annotation)
    data.append(f'# ***annotation***\n# {anno_text}\n# ***annotation***\n')
    with open(fn, 'w') as f:
        f.writelines(data)
    return fn, annotation

def _annotate_xyz(fn, annotation: dict):
    '''
    Annotate the structure file with additional information

    Parameters
    ----------
    fn : str
        The file name of the structure file
    annotation : dict
        The annotation information
    '''
    with open(fn, 'r') as f:
        data = f.readlines()
    anno_text = _dict_as_text(annotation)
    data[1] = f'{anno_text}\n'
    with open(fn, 'w') as f:
        f.writelines(data)
    return fn, annotation

def _annotate_deform(fn, fmt, annotation: dict):
    '''
    Write the annotation to file

    Parameters
    ----------
    fn : str
        The file name of the annotation file
    fmt : str
        The format of the annotation file
    annotation : dict
        The annotation information
    '''
    if fmt == 'abacus/stru':
        return _annotate_abacus_stru(fn, annotation)
    elif fmt == 'xyz':
        return _annotate_xyz(fn, annotation)
    else:
        raise NotImplementedError(f'Unsupported format: {fmt}')
    
def write_pymatgen_deform(obj: DeformedStructureSet, 
                          fmt = 'abacus/stru', 
                          outdir = None, 
                          prefix = 'elastic'):
    '''
    Export the deformed structure to file
    '''
    logging.info(f'Exporting deformed structure to file, format: {fmt}, prefix: {prefix} >>')
    out = []
    outdir = outdir if outdir else os.getcwd()
    temp = pymatgen2dpdata(obj)

    for i, (s, desc) in enumerate(temp):
        fn = os.path.join(outdir, f'STRU-{prefix}-{i}')
        s.to(fmt, fn)
        _ = _annotate_deform(fn, fmt, desc)
        out.append(fn)
        
    logging.info(f'<< Exported deformed structure to file. In total: {len(out)}')
    return out

def read_stress(fn, unit = 'GPa'):
    '''read the stress tensor from abacus running_*.log file'''
    stress = read_abacus_stress(fn)
    return Stress(-1 * _convert_pressure_unit(stress, 'kbar', unit))

def _abacus_shortcut(workdir, unit = 'GPa'):
    '''a short cut, if the ABACUS run has been done before,
    the calculation will be skipped'''
    if os.path.exists(os.path.join(workdir, 'OUT.ABACUS', 'running_scf.log')):
        return read_stress(os.path.join(workdir, 'OUT.ABACUS', 'running_scf.log'), unit)
    else:
        return None

def _correct_abacus_stru(fn, fpsp, forb):
    '''correct the ABACUS STRU file in case it is not properly written'''
    # cut the file into pieces, the first is all below the LATTICE_CONSTANT (inclusive)
    # the second is all below the ATOMIC_POSITIONS (exclusive)
    # this function will read the element symbol from the second part
    # then rewrite the part above the first part
    fpsp = [fpsp] if isinstance(fpsp, str) else fpsp
    forb = [forb] if isinstance(forb, str) else forb
    
    fpsp = [os.path.basename(f) for f in fpsp]
    forb = [os.path.basename(f) for f in forb]
    
    with open(fn, 'r') as f:
        data = f.readlines()
    data = [l.split('#')[0].strip() + '\n' for l in data] # remove comments
    
    second, first = 0, 0
    for i, line in enumerate(data):
        if line.strip().startswith('ATOMIC_POSITIONS'):
            second = i
        if line.strip().startswith('LATTICE_CONSTANT'):
            first = i
    if second == 0 or first == 0:
        raise ValueError('Cannot find the ATOMIC_POSITIONS or LATTICE_CONSTANT')
    
    # get the element symbol from the second part
    elem = re.findall(r'([A-Z][a-z]) \d+(\.\d+)? \d', ' '.join([line.strip() for line in data[second:]]))
    elem = [e[0] for e in elem]
    if len(elem) == 0:
        raise ValueError('Cannot find the element symbol')
    
    # rewrite the first part
    header = 'ATOMIC_SPECIES\n'
    for e, f in zip(elem, fpsp):
        header += f'{e} 1.0000 {f}\n'
    header += '\nNUMERICAL_ORBITAL\n'
    header += '\n'.join(forb) + '\n\n'
    
    with open(fn, 'w') as f:
        f.writelines(header)
        f.writelines(data[first:])
    return fn

def abacus_calculator(calculator, fn ,**kwargs):
    '''calculate the stress with ABACUS
    
    Parameters
    ----------
    calculator : str
        The command to run ABACUS
    fn : str
        The file name of the structure file
    fdft : str
        The file name of the ABACUS INPUT file
    fpsp : str|list
        The file name of the pseudopotential file
    forb : str|list
        The file name of the orbital file
    
    Returns
    -------
    pymatgen.analysis.elasticity.stress.Stress
        The stress tensor       
    '''
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
    # the ATOMIC_SPECIES and NUMERICAL_ORBITAL section may not be properly written
    # here we cut the file and write them again
    _correct_abacus_stru(os.path.join(workdir, 'STRU'), fpsp, forb)
        
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
    cal_type = dft.get('calculation', 'scf')
    fout = os.path.join(workdir, f'OUT.{suffix}', f'running_{cal_type}.log')
    return read_stress(fout, kwargs.get('unit', 'GPa'))

def dp_calculator(calculator, fn, **kwargs):
    '''calculate the stress with Deltaspin model'''
    raise NotImplementedError('Calculation of stress with Deltaspin model is not implemented yet')

def cal_stress(calculator, fn, fmt = 'abacus/stru', **kwargs):
    '''calculate the stress of the given structure'''
    logging.info(f'Calculating stress with {calculator} >>')
    
    out = None
    if calculator.endswith('.pth'): # Deltaspin model
        out = dp_calculator(calculator, fn, **kwargs)
    elif 'abacus' in calculator: # ABACUS DFT
        out = abacus_calculator(calculator, fn, **kwargs)
    else:
        raise ValueError(f'Unsupported calculator: {calculator}')

    logging.info(f'<< Calculated stress with {calculator}')
    return out

def read_deform_as_dpdata(fn: str|list):
    '''
    the counterpart of `write_pymatgen_deform`
    
    Parameters
    ----------
    fn : str|list
        The file name of the deformed structure file, or a list of file names
    
    Returns
    -------
    list of tuple of dpdata.System and dict
        The dpdata.System object and the deformation information
    '''
    out = []
    fn = fn if isinstance(fn, list) else [fn]
    for fabacus in fn:
        with open(fabacus) as f:
            data = f.read()
        anno = data.split('# ***annotation***\n')[1][2:] # remove the first '# '
        deform = _text_as_dict(anno)
        out.append((dpdata.System(fabacus, fmt = 'abacus/stru'), deform))
    return out

def calculate(strain_and_stress, stress0, unit = 'GPa'):
    '''
    calculate various elastic coefficients with given strains and stresses

    Parameters
    ----------
    strain_and_stress : list of tuple of strain and 
    pymatgen.analysis.elasticity.stress.Stress object
        The strain and stress information of each deformed structure
    stress0 : pymatgen.analysis.elasticity.stress.Stress object
        The stress information of the original structure (before deformation)
    
    Returns
    -------
    dict
        The elastic properties, including elastic tensor, bulk modulus,
        shear modulus, Young's modulus, Poisson's ratio, and the stress
        tensor of the original structure
    dict
        The vocabulary of the full name
    dict
        The units of the elastic properties
    '''
    logging.info(f'Calculating elastic properties >>')
    strain, stress = zip(*strain_and_stress)
    result = ElasticTensor.from_independent_strains(strain, 
                                                    stress,
                                                    eq_stress=stress0,
                                                    vasp=False)
    # will also return the vocabulary of the units and the description
    vocabularies = {'C': 'elastic tensor', 'BV': 'bulk modulus', 'GV': 'shear modulus',
                    'EV': "Young's modulus", 'uV': "Poisson's ratio"}
    units = {'C': unit, 'BV': unit, 'GV': unit, 'EV': unit, 'uV': None}
    bv, gv = result.k_voigt, result.g_voigt
    logging.info(f'<< Calculated elastic properties')
    return {
        'C': np.array([result.voigt[i][j] \
                       for i in range(6) \
                       for j in range(6)]).reshape(6, 6),
        'BV': bv,
        'GV': gv,
        'EV': 9 * bv * gv / (3 * bv + gv),
        'uV': (3 * bv - 2 * gv) / (6 * bv + 2 * gv)
    }, vocabularies, units

def main(calculator, fn, norm=1e-2, shear=1e-2, **kwargs):
    '''
    The main function to calculate the elastic properties

    Parameters
    ----------
    calculator : str
        The command to run ABACUS, or the file name of the Deltaspin model
    fn : str
        The file name of the structure file
    fdft : str
        The file name of the ABACUS INPUT file
    fpsp : str|list
        The file name of the pseudopotential file
    forb : str|list
        The file name of the orbital file
    norm : float
        The norm strain, default is 1e-2
    shear : float
        The shear strain, default is 1e-2
    unit : str
        The unit of the elastic properties, default is 'GPa'
    
    
    Returns
    -------
    dict
        The elastic properties, including elastic tensor, bulk modulus,
        shear modulus, Young's modulus, Poisson's ratio, and the stress
        tensor of the original structure
    dict
        The vocabulary of the full name
    dict
        The units of the elastic properties
    '''
    # standardize the structure at the very beginning
    fn = ieee_structure(fn)
    # read the original structure
    stru = read_structure(fn)
    # calculate stress0
    stress0 = cal_stress(calculator, fn, **kwargs)
    # deform the structure
    deformed = deform(stru, norm = norm, shear = shear)
    # write the deformed structure to file
    structures = write_pymatgen_deform(deformed)
    # calculate the stress of deformed structures
    strain = [Strain.from_deformation(d) for d in deformed.deformations]
    stress = [cal_stress(calculator, fdfm, **kwargs) for fdfm in structures]
    # calculate the elastic properties
    return calculate(list(zip(strain, stress)), stress0)

class TestElastic(unittest.TestCase):

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

    def test_read_structure(self):
        '''
        Test `read_structure` function
        '''
        fn = self.init_mytest()
        temp = read_structure(fn)
        os.remove(fn)
        self.assertTrue(isinstance(temp, Structure))
        self.assertEqual(len(temp.sites), 2)

    def test_deform(self):
        '''
        Test `deform` function
        '''
        fn = self.init_mytest()
        stru_obj = read_structure(fn)
        os.remove(fn)
        deformed = deform(stru_obj)
        self.assertEqual(len(deformed), 24) # will generate 24 deformed structures
        self.assertTrue(isinstance(deformed, DeformedStructureSet))

    @unittest.skip('this is not really a unittest, but to ensure the function works')
    def test_export(self):
        '''
        Test `export` function
        '''
        fn = self.init_mytest()
        stru_obj = read_structure(fn)
        os.remove(fn)
        deformed = deform(stru_obj)
        write_pymatgen_deform(deformed)

    def test_correct_abacus_stru(self):
        '''
        Test `_correct_abacus_stru` function
        '''
        fn = self.init_mytest()
        fpsp = 'Fe.upf'
        forb = 'Fe_gga_7au_100Ry_4s2p2d1f.orb'
        _correct_abacus_stru(fn, fpsp, forb)
        
        with open(fn, 'r') as f:
            data = f.read()
        self.assertTrue('Fe.upf' in data)
        self.assertTrue('Fe_gga_7au_100Ry_4s2p2d1f.orb' in data)
        
        os.remove(fn)

    def test_make_primitive(self):
        '''
        Test `_make_primitive` function
        '''
        fn = self.init_mytest()
        stru_obj = read_structure(fn)
        os.remove(fn)
        # get its primitive cell
        stru_obj = _make_primitive(stru_obj)
        # because the fn contains two atoms defining BCC, but BCC actually
        # has only 1 atom in primitive cell, so the number of atoms should be 1
        self.assertEqual(len(stru_obj.sites), 1)

    def test_ieee_structure(self):
        '''
        Test `ieee_structure` function
        '''
        fn = self.init_mytest()
        fnew = ieee_structure(fn)
        os.remove(fn)
        self.assertTrue(os.path.exists(fnew))
        os.remove(fnew)

if __name__ == '__main__':
    flog = f'elastic-{time.strftime("%Y%m%d-%H%M%S")}.log'
    logging.basicConfig(filename = flog, level = logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    test = init()
    unittest.main(exit=test)

    out = main(
        # basic structure definition
        # --------------------------
        fn='phonopy-Diamond-Si/STRU',
        
        # stress calculator
        # -----------------
        calculator='mpirun -np 16 abacus',
        
        # strain magnitude
        # ----------------
        norm=1e-2,
        shear=1e-2,

        # DFT additional parameters
        # -------------------------
        fdft='phonopy-Diamond-Si/INPUT',
        fpsp='pp_orb/Si_ONCV_PBE-1.0.upf',
        forb='pp_orb/Si_gga_10au_100Ry_2s2p1d.orb'
        )
    
    data, vocab, units = out
    
    print('Properties output:\nElastic tensor:')
    for d in data['C']:
        print(' '.join([f'{v:7.2f}' for v in d]))
    print(f'Bulk modulus: {data["BV"]:.2f} GPa')
    print(f'Shear modulus: {data["GV"]:.2f} GPa')
    print(f"Young's modulus: {data['EV']:.2f} GPa")
    print(f"Poisson's ratio: {data['uV']:.2f}")

    logging.shutdown()
    print(f'Log file: {flog}')
