'''
deltaspin basic module

Version
-------
2024/12/23 14:30

Prerequisites
-------------
install dpdata the devel version from github
```
    git clone https://github.com/deepmodeling/dpdata.git
    cd dpdata
    git checkout devel
    pip install .
```

Extension
---------
1. implement new perturbation types
   see functions _pert_cell and _pert_magmom
2. add your new perturbation type to pert() function
'''
# external modules
import dpdata
import numpy as np

# internal modules
import copy
import os
import itertools as it
import unittest
import time
import logging
import uuid
import shutil

# home-made modules
from abacus import read_stru
from utils import init

def _deepmd_signature(fn, fmt='abacus/stru'):
    '''generate the deepmd signature of the structure file'''
    obj = dpdata.System(fn, fmt=fmt)
    # the signature is the chemical composition notation
    return ''.join([f'{e}{n}' for e, n in zip(obj.get_atom_names(), obj.get_atom_numbs())])

def is_dpdata_dir(root):
    '''check whether the root is a dpdata directory'''
    return os.path.exists(os.path.join(root, 'type.raw')) and \
           os.path.exists(os.path.join(root, 'type_map.raw'))

def read_forces(fn, jobdir):
    '''read the atomic forces
    
    Parameters
    ----------
    fn : str
        file name of the forces
    jobdir : str
        job directory
    
    Returns
    -------
    list
        (elem, forces) pairs of all frames
    '''
    logging.info(f'Read forces from {fn} in {jobdir} >>')
    errmsg = None
    if not os.path.exists(fn):
        errmsg = f'{fn} not found'
    if not os.path.exists(jobdir):
        errmsg = f'{jobdir} not found'
    if not os.path.exists(os.path.join(jobdir, 'type_map.raw')):
        errmsg = f'{os.path.join(jobdir, "type_map.raw")} not found'
    if not os.path.exists(os.path.join(jobdir, 'type.raw')):
        errmsg = f'{os.path.join(jobdir, "type.raw")} not found'
    if errmsg is not None:
        logging.error(errmsg)
        raise FileNotFoundError(errmsg)
    # the force read from the file dumped by dp_calculator will have
    # six columns, the last three are the forces predicted
    forces = np.loadtxt(fn, skiprows=1)[:, -3:].astype(float)
    # in jobdir, read the type_map.raw and type.raw
    type_map = np.loadtxt(os.path.join(jobdir, 'type_map.raw'), dtype=int)
    types = np.loadtxt(os.path.join(jobdir, 'type.raw'), dtype=str)
    
    # the forces are written in the order of type_map, and all frames
    # are successively written, so we need to distinguish the frames
    nat = len(type_map)
    forces = forces.reshape(-1, nat, 3)
    elem = [types[i] for i in type_map]
    
    logging.info(f'<< Read forces from {fn} in {jobdir}')
    return [(elem, i) for i in forces]

def write_forces(prefix, forces, outdir = None, proj = None, mag_force = False):
    '''write the forces to file in dpdata dumped format
    
    Parameters
    ----------
    fn : str
        file name
    forces : np.ndarray
        forces
    outdir : str
        output directory, default is None
    proj : str
        project name, can be None
    
    Returns
    -------
    str
        file name
    '''
    # file I/O control
    outdir = outdir if outdir else os.getcwd()
    fn = f'{prefix}.'
    fn += 'fm.out' if mag_force else 'fr.out'
    fn = os.path.join(outdir, fn)
    
    header = '# '
    if proj is not None:
        header += proj + ': '
    header += 'data_fx data_fy data_fz pred_fx pred_fy pred_fz'
    # no validation, no reference data, so the first three numbers are always 0

    forces = forces.reshape(-1, 3) # force in format of <24.18e
    # add three zero columns
    forces = np.hstack((np.zeros((forces.shape[0], 3)), forces))
    np.savetxt(fn, forces, fmt='%24.18e', header=header, comments='')
    return fn

def write_energies(prefix, energies, outdir = None, proj = None, peratom = False):
    '''write the energies to file in dpdata dumped format'''
    # file I/O control
    outdir = outdir if outdir else os.getcwd()
    fn = f'{prefix}.'
    fn += 'e_peratom.out' if peratom else 'e.out'
    fn = os.path.join(outdir, fn)
    
    header = '# '
    if proj is not None:
        header += proj + ': '
    header += 'data_e pred_e'
    # no validation, no reference data, so the first one number are always 0

    energies = energies.reshape(-1, 1) # energy in format of <24.18e
    # add one zero column
    energies = np.hstack((np.zeros((energies.shape[0], 1)), energies))
    np.savetxt(fn, energies, fmt='%24.18e', header=header, comments='')
    return fn

def write_virials(prefix, virials, outdir = None, proj = None, peratom = False):
    '''write the virials to file in dpdata dumped format'''
    # file I/O control
    outdir = outdir if outdir else os.getcwd()
    fn = f'{prefix}.'
    fn += 'v_peratom.out' if peratom else 'v.out'
    fn = os.path.join(outdir, fn)
    
    header = '# '
    if proj is not None:
        header += proj + ': '
    header += 'data_vxx data_vxy data_vxz data_vyx data_vyy data_vyz data_vzx data_vzy \
data_vzz pred_vxx pred_vxy pred_vxz pred_vyx pred_vyy pred_vyz pred_vzx pred_vzy pred_vzz'

    # no validation, no reference data, so the first nine numbers are always 0
    virials = virials.reshape(-1, 9) # virial in format of <24.18e
    # add nine zero columns
    virials = np.hstack((np.zeros((virials.shape[0], 9)), virials))
    np.savetxt(fn, virials, fmt='%24.18e', header=header, comments='')
    return fn

def _field_selector(identifiers, field: str):
    '''
    return the indexes of atoms in the system specified by the field

    Parameters
    ----------
    identifiers : np.ndarray
        identifier of each atom
    field : str
        can be atom:1, which means the first atom, or type:Fe, which means all Fe atoms
        or type:all or atom:all, which means all atoms
    
    Returns
    -------
    out : np.ndarray
        indexes of atoms
    '''
    def _parse_range(pattern):
        for pat in pattern.split(','):
            if '-' in pat:
                start, end = pat.split('-')
                yield from range(int(start) - 1, int(end))
            else:
                yield int(pat) - 1 # because the index actually starts from 0

    identifiers = np.array(identifiers) if isinstance(identifiers, list) else identifiers
    nat, = identifiers.shape

    field = 'atom:all' if field is None else field

    if field == 'atom:all':
        return np.arange(nat)
    elif field.startswith('atom:'): # handles like atom:1, atom:1-3, atom:1,3-5,10
        return np.array(list(_parse_range(field.split(':')[1])))
    elif field.startswith('type:'):
        ityp = list(_parse_range(field.split(':')[1]))
        return np.array(sorted([j for i in ityp for j in np.where(identifiers == i)[0]]))
    else:
        errmsg = f'Invalid field selector: {field}'
        logging.error(errmsg)
        raise ValueError(errmsg)

def _pert_cell(proto: dpdata.System, scale: float):
    '''scale the cell volume isotropically by a factor of (1+mag)**(1/3)'''
    c = proto.data['cells'][0]
    tau_c = proto.data['coords'][0]
    # solve to obtain the Direct coordinates
    tau_d = np.linalg.solve(c.T, tau_c.T).T
    # scale the cell
    c *= (1 + scale)**(1/3)
    # recover the Cartesian coordinates
    tau_c = np.dot(tau_d, c)

    out = copy.deepcopy(proto)
    out.data['cells'] = np.array([c])
    out.data['coords'] = np.array([tau_c])
    return out

def _pert_magmom(proto: dpdata.System, mag, overwrite = False, field = None):
    '''set the magnetic moment of each atom to mag
    
    Parameters
    ----------
    proto : dpdata.System
        prototype
    mag : float or np.ndarray
        magnetic moment
    field : str
        can be atom:1, which means the first atom, or type:Fe, which means all Fe atoms
        default is None, which means all atoms

    Returns
    -------
    out : dpdata.System
        perturbed system
    '''
    nat = len(proto.data['coords'][0])
    idx = _field_selector(proto.data['atom_types'], field)
    
    # confusing! seems the `spins` cannot be read-in by dpdata
    spins = proto.data.get('spins', [np.zeros((nat, 3))])[0]
    if isinstance(overwrite, np.ndarray):
        # seems I have to implement this...
        if overwrite.shape == (3,): # one for all
            spins = np.array([overwrite] * nat)
        elif overwrite.shape == (nat, 3): # one for each
            spins = overwrite
        else:
            errmsg = f'Too complicated! This functionality is not fully implemented yet: {overwrite.shape}'
            logging.error(errmsg)
            raise ValueError(errmsg)
    elif overwrite:
        if isinstance(overwrite, bool):
            spins = np.array([[0, 0, 1]] * nat)
        elif isinstance(overwrite, (float, int)):
            spins = np.array([[0, 0, overwrite]] * nat)
        else:
            errmsg = f'Invalid overwrite type: {type(overwrite)}'
            logging.error(errmsg)
            raise ValueError(errmsg)

    if isinstance(mag, float):
        spins[idx, 2] = mag
    elif isinstance(mag, np.ndarray):
        # the mag here can only be a np.ndarray with shape
        # (3,): one for all selected by idx
        # (N, 3): for each atom selected by idx
        spins[idx] = mag
    else:
        errmsg = f'Invalid type of mag: {type(mag)}'
        logging.error(errmsg)
        raise ValueError(errmsg)

    out = copy.deepcopy(proto)
    out.data['spins'] = np.array([spins])
    return out

def pert(proto, overwrite, **kwargs):
    '''apply perturbations to the prototype'''
    out = copy.deepcopy(proto)
    for key, val in kwargs.items():
        if key == 'cell':
            out = _pert_cell(out, val)
        elif key.startswith('magmom'):
            field = 'atom:all' if key == 'magmom' else key.replace('magmom:', '')
            out = _pert_magmom(out, val, 
                               overwrite=overwrite, 
                               field=field)
        else:
            errmsg = f'Unknown perturbation type: {key}'
            logging.error(errmsg)
            raise ValueError(errmsg)
    return out

def predict(fmodel, testdir, prefix):
    '''predict by model on systems in testdir. Output data in outdir
    
    Parameters
    ----------
    fmodel : str
        model file
    testdir : str
        test data directory
    prefix : str
        The prefix to files where details of energy, force and virial 
        accuracy/accuracy per atom will be written
    '''
    logging.info(f'Predict with model: {fmodel} >>')
    # -m: model file
    # -s: test data directory
    # -d: prefix
    msg = f'''Run deepmd-kit to predict properties of system:
model: {fmodel}
test data directory: {testdir}
prefix: {prefix}'''
    logging.info(msg)
    os.system(f'dp --pt test -m {fmodel} -s {testdir} -d {prefix}')
    logging.info('<< Predicted.')

def main(fn, 
         fmt,
         pertkinds, 
         pertmags, 
         jobdir,
         out_fmt='deepmd/npy',
         overwrite=False,
         fmodel=None,
         prefix='out'):
    '''
    main workflow function

    Parameters
    ----------
    fn : str
        filename
    fmt : str
        format
    pertkinds : list[str]
        perturbation kinds
    pertmags : list[Iterable[float]]
        perturbation magnitudes
    jobdir : str
        job directory storing the generated data to be predicted
    out_fmt : str
        output format, default is deepmd/npy
    overwrite : bool|float|np.ndarray
        overwrite the magnetic moments of all the rest atoms
    fmodel : str
        model file
    prefix : str
        prefix of the output files
    '''
    # generate
    proto = dpdata.System(fn, fmt=fmt)
    out = dpdata.MultiSystems()
    for mag in it.product(*pertmags):
        out.append(pert(proto, overwrite, **dict(zip(pertkinds, mag))))
    out.to(out_fmt, jobdir)

    # predict
    print(f'Do prediction on systems generated ({jobdir})?: {fmodel is not None}')
    if fmodel:
        logging.info(f'predicting by model: {fmodel}, on systems {jobdir}')
        predict(fmodel, jobdir, prefix)

def to_abacus(root, outdir, spin_constrain = False, **kwargs):
    '''convert all the deepmd data in root to abacus format
    
    Parameters
    ----------
    root : str
        root directory of one system dumped by dpdata. In some contexts, it
        is not the `jobdir`, but the subfolder of it.
    outdir : str
        output directory
    spin_constrain : bool
        whether to constrain the spin. Default is False, if set to True, there
        will be additional `sc 1 1 1` be written at the end of each line defining
        the atom
    fpsp : list[str], optional
        list of pseudopotential files. If not provided, this will be left empty
        for all elements
    forb : list[str], optional
        list of numerical orbital files. If not provided, the section 
        `NUMERICAL_ORBITAL` will not be written
    
    Returns
    -------
    list[str]
        list of filenames of the converted structures
    '''
    logging.info(f'Convert all the deepmd data in {root} to abacus format >>')
    if not os.path.exists(root):
        errmsg = f'{root} not found'
        logging.error(errmsg)
        raise FileNotFoundError(errmsg)

    ftyp = os.path.join(root, 'type.raw')
    ftyp_map = os.path.join(root, 'type_map.raw')
    if not os.path.exists(ftyp) or not os.path.exists(ftyp_map):
        errmsg = f'{ftyp} or {ftyp_map} not found'
        logging.error(errmsg)
        raise FileNotFoundError(errmsg)

    with open(ftyp) as f:
        typ = f.readlines()
    typ = [t.strip() for t in typ]
    typ = [int(t) for t in typ if t]
    nat = len(typ)

    if os.path.exists(outdir):
        #raise FileExistsError(f'{outdir} already exists')
        logging.warning(f'{outdir} already exists, will be overwritten')
    else:
        os.makedirs(outdir)
    
    systems = dpdata.System(root, fmt='deepmd/npy')
    options = {'pp_file': kwargs.get('fpsp'),
               'numerical_orbital': kwargs.get('forb')}
    if spin_constrain:
        options |= {'sc': [[1, 1, 1]] * nat}
    # if spin-constrain
    fn = [os.path.join(outdir, f'{i}.stru') for i in range(systems.get_nframes())]
    for i, f in enumerate(fn):
        systems.to('abacus/stru', f, frame_idx=i, **options)
    return fn

class DeltaspinTest(unittest.TestCase):
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
    
    def test_deepmd_signature(self):
        '''test _deepmd_signature'''
        fn = self.init_mytest()
        out = _deepmd_signature(fn)
        os.remove(fn)
        self.assertEqual(out, 'Fe2')
        
    def test_field_selector(self):

        identifiers = [0, 1, 0, 2, 1, 0]
        self.assertEqual(_field_selector(identifiers, 'atom:all').tolist(), [0, 1, 2, 3, 4, 5])
        self.assertEqual(_field_selector(identifiers, 'atom:1').tolist(), [0])
        self.assertEqual(_field_selector(identifiers, 'atom:1,3-5').tolist(), [0, 2, 3, 4])
        self.assertEqual(_field_selector(identifiers, 'type:1').tolist(), [0, 2, 5])
        self.assertEqual(_field_selector(identifiers, 'type:2').tolist(), [1, 4])
        self.assertEqual(_field_selector(identifiers, 'type:3').tolist(), [3])
        self.assertEqual(_field_selector(identifiers, 'type:1,3').tolist(), [0, 2, 3, 5])
        self.assertEqual(_field_selector(identifiers, 'type:1-2').tolist(), [0, 1, 2, 4, 5])

    @unittest.skip('This is merely an example, showing how this code works')
    def test_integrated(self):
        '''This is merely an example, showing how this code works'''
        proto = 'BCC-Fe'
        overwrite = 2.4
        fmt = 'abacus/stru'
        fmodel = 'pw-bcc-fm-dpa1-iter11-20241213/model.pth'
        prefix = 'feel_lucky'
        jobdir = prefix

        mags = [np.array([m, 0, 0]) for m in np.linspace(1, 3, 31, endpoint=True)]
        cell = [0]

        pertkinds = ['cell', 'magmom:atom:1']
        pertmags = [cell, mags]

        flog = f'DeltaspinPESGenerator@{time.strftime("%Y%m%d-%H%M%S")}.log'
        jobdir = jobdir if jobdir \
            else f'DeltaspinPESGeneratorJob@{time.strftime("%Y%m%d-%H%M%S")}'
        logging.basicConfig(filename=flog, 
                            level=logging.INFO, 
                            format='%(asctime)s - %(message)s')
        # astonishing! cp2kdata overwrited the logging???
        logging.info(f'log file: {flog}')
        logging.info(f'generated jobs will be in folder: {jobdir}')

        main(fn=proto, 
             fmt=fmt, 
             pertkinds=pertkinds,
             pertmags=pertmags, 
             jobdir=jobdir,
             overwrite=overwrite,
             fmodel=fmodel,
             prefix=prefix)

        logging.info('done')
        logging.shutdown()
    
    def test_to_abacus(self):
        
        typ = [0, 1, 0, 2, 1, 0]
        typ_map = ['H', 'C', 'O']
        box = [np.random.rand(3, 3).flatten() for _ in range(6)]
        coord = [np.random.rand(6, 3).flatten() for _ in range(6)]
        # 1 frame, 6 atoms, saved in flattened form
        # save in deepmd format, the coord.npy will be in the folder
        # set.000
        temp = f'test_to_abacus_{str(uuid.uuid4())}'
        os.makedirs(temp)
        with open(os.path.join(temp, 'type.raw'), 'w') as f:
            f.write('\n'.join([str(t) for t in typ]))
        with open(os.path.join(temp, 'type_map.raw'), 'w') as f:
            f.write('\n'.join(typ_map))
        os.makedirs(os.path.join(temp, 'set.000'))
        np.save(os.path.join(temp, 'set.000', 'coord.npy'), np.array(coord))
        np.save(os.path.join(temp, 'set.000', 'box.npy'), np.array(box))
        # convert to abacus format
        outdir = f'test_to_abacus_{str(uuid.uuid4())}'
        fn = to_abacus(temp, outdir)
        self.assertEqual(len(fn), 6) # because there are 6 frames
        for i in range(6):
            self.assertTrue(os.path.exists(fn[i]))
        
        coord = np.array(coord).reshape(6, -1, 3) # 6 frames, 6 atoms, 3 coordinates
        box = np.array(box).reshape(6, 3, 3) # 6 frames, 3*3 components
        for i in range(6):
            stru = read_stru(fn[i])
            self.assertEqual(len(stru['ATOMIC_SPECIES']), 3)
            for j in range(3): 
                # each mass is 1 because it is not set, 
                # len is 2 because fpsp is not set
                self.assertEqual(stru['ATOMIC_SPECIES'][j][1], '1')
                self.assertEqual(len(stru['ATOMIC_SPECIES'][j]), 2)
            lat = np.array([float(x) for l in stru['LATTICE_VECTORS'] for x in l]).reshape(3, 3)
            self.assertTrue(np.allclose(lat, box[i]))
            self.assertEqual(stru['LATTICE_CONSTANT'][0][0], '1.8897261246257702')
            # so also three groups of atoms
            self.assertEqual(len(stru['ATOMIC_POSITIONS']['atoms']), 3)
            # 3 Hydrogen, 2 Carbon, 1 Oxygen
            self.assertEqual(stru['ATOMIC_POSITIONS']['atoms'][0]['nat'], 3)
            self.assertEqual(stru['ATOMIC_POSITIONS']['atoms'][1]['nat'], 2)
            self.assertEqual(stru['ATOMIC_POSITIONS']['atoms'][2]['nat'], 1)
    
        shutil.rmtree(temp)
        shutil.rmtree(outdir)
    
    def test_write_forces(self):
        forces = np.random.rand(6, 3)
        fn = write_forces('test_write_forces', forces)
        self.assertTrue(os.path.exists(fn))
        with open(fn) as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 6 + 1)
        for i in range(6):
            self.assertTrue(lines[i + 1].startswith('0.000000000000000000e+00'))
        os.remove(fn)
    
if __name__ == '__main__':

    test = init()
    unittest.main(exit=test)
    out = to_abacus('magmom-exch-const-test/Fe2', 'magmom-exch-const-test/Fe2-abacus', True, 
                    fpsp = ['Fe.pbe-spn-rrkjus_psl.1.0.0.UPF'],
                    forb = ['Fe_gga_10au_100Ry_4s2p2d1f.orb'])
    print(out)
    