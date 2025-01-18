'''
this file contains common utility functions that are used in the project
'''
import numpy as np
import logging
import re
import uuid
import unittest
import os
import shutil

from utils import _listdict2dictlist, convert_force_unit, convert_energy_unit, init
from structure import _build_supercell

def read_fdft(fn):
    '''get some general information from ABACUS INPUT'''
    with open(fn) as f:
        data = f.readlines()
    data = [line.strip().split('#')[0] for line in data]
    data = [l for l in data if l] # remove empty lines
    out = {}
    for l in data:
        words = l.split()
        k = words[0]
        v = ' '.join(words[1:])
        out[k] = v
    return out

def _kmeshgen(cell, kspacing):
    """get the Monkhorst-Pack mesh
    This function is copied from: https://github.com/kirk0830/ABACUS-Pseudopot-Nao-Square/blob/58d4693b9e9c7567c42907bdd8727f59533c8021/apns/test/atom_species_and_cell.py#L366
    """
    import numpy as np
    kspacing = kspacing if isinstance(kspacing, list) else [kspacing] * 3
    vecs = np.array(cell).reshape(3, 3) # no matter what the shape is of the cell
    recvecs = np.linalg.inv(vecs).T
    recvecs = 2*np.pi * recvecs
    norms = np.linalg.norm(recvecs, axis=1).tolist()
    assert len(norms) == len(kspacing), f'kspacing should be a list of 3 floats: {kspacing}'
    norms = [int(norm / kspac) for norm, kspac in zip(norms, kspacing)]
    return list(map(lambda x: max(1, x + 1), norms))

def _write_abacus_kpt(fn, nk, center = 'Gamma', kshift = [0, 0, 0]):
    '''write the KPT file for ABACUS'''
    if isinstance(nk, int):
        nk = [nk] * 3
    if not isinstance(nk, list) or len(nk) != 3:
        raise ValueError('nk must be a list of 3 integers')

    center = 'Gamma' if center.lower() == 'gamma' else 'MP'
    out = f'''K_POINTS\n0\n{center}\n'''
    out += ' '.join([str(n) for n in nk + kshift]) + '\n'
    with open(fn, 'w') as f:
        f.write(out)
    return fn

def read_stress(fn):
    '''
    read the stress tensor from the output file of ABACUS
    '''
    with open(fn) as f:
        data = f.readlines()
    # the stress tensor will appear in the next 4 lines after the title line
    # "TOTAL-STRESS (KBAR)", but the 1st line is trash
    stress = None
    for i, line in enumerate(data):
        if 'TOTAL-STRESS (KBAR)' in line:
            stress = data[i+2:i+5]
    if not stress:
        raise ValueError('Cannot find the stress tensor')
    stress = np.array([float(s) for s in ' '.join(stress).split()]).reshape(3, 3)
    # multiply -1 because the difference between `in` and `out`
    return stress

def _write_cp2k_forces(fn, elem, forces, unit = 'eV/Angstrom'):
    '''write forces to cp2k format
    
    Parameters
    ----------
    fn : str
        output file name
    elem : list
        list of elements, of each atom
    forces : list or np.ndarray
        atomic forces, should have size in total as 3 * len(elem)
    unit : str
        unit of the forces input, default is 'eV/Angstrom'. will convert
        to a.u. (because CP2K uses a.u. for forces)
    
    Returns
    -------
    fn : str
        output file name
    '''
    out = '''
ATOMIC FORCES in [a.u.]

 # Atom   Kind   Element          X              Y              Z
'''
    # convert the forces to a.u. and reshape
    forces = np.array(forces).flatten()
    forces = np.array([convert_force_unit(i, unit, 'a.u.') for i in forces]).reshape(-1, 3)
    
    if len(elem) != len(forces):
        raise ValueError("The length of elem and forces should be the same")
    
    # build a type map
    types = list(dict.fromkeys(elem))
    type_map = [types.index(i) for i in elem]
    
    for i, (e, it, f) in enumerate(zip(elem, type_map, forces)):
        j, jt = i + 1, it + 1 # fortran starts from 1
        out += f' {j:>6d} {jt:>6d} {e:>7s}      {f[0]:>14.8f} {f[1]:>14.8f} {f[2]:>14.8f}\n'
    # then the total forces in X, Y, Z direction, and the total force
    total_force = np.sum(forces, axis=0)
    out += f' SUM OF ATOMIC FORCES       {total_force[0]:>14.8f} {total_force[1]:>14.8f} {total_force[2]:>14.8f}'
    out += f' {np.linalg.norm(total_force):>19.8f}\n'
    
    with open(fn, 'w') as f:
        f.write(out)
    return fn

def write_forces(fn, elem, forces):
    '''write the forces to the ABACUS running_*.log format
    
    Parameters
    ----------
    fn : str
        output file name
    elem : list
        list of elements, of each atom
    forces : list or np.ndarray
        atomic forces, should have size in total as 3 * len(elem)
    '''
    # the format of forces in ABACUS running_*.log is:
    # {elem:>2s}{idx:<4d}{f[0]:>20.10f}{f[1]:>20.10f}{f[2]:>20.10f}
    
    out = '''
                ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡟⠋⠈⠙⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠤⢤⡀⠀⠀
                ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠈⢇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠞⠀⠀⢠⡜⣦⠀
                ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡃⠀⠀⠀⠀⠈⢷⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠊⣠⠀⠀⠀⠀⢻⡘⡇
                ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠃⠀⠀⠀⠀⠀⠀⠙⢶⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠⠚⢀⡼⠃⠀⠀⠀⠀⠸⣇⢳
                ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⠀⣀⠖⠀⠀⠀⠀⠉⠀⠀⠈⠉⠛⠛⡛⢛⠛⢳⡶⠖⠋⠀⢠⡞⠀⠀⠀⠐⠆⠀⠀⣿⢸
                ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠻⣦⣀⣴⡟⠀⠀⢶⣶⣾⡿⠀⠀⣿⢸
                ⠀⠀⠀⠀⠀⠀⠀⠀⢀⣤⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀⡠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣏⠀⠀⠀⣶⣿⣿⡇⠀⠀⢏⡞
                ⠀⠀⠀⠀⠀⠀⢀⡴⠛⠀⠀⠀⠀⠀⠀⠀⠀⢀⢀⡾⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢦⣤⣾⣿⣿⠋⠀⠀⡀⣾⠁
                ⠀⠀⠀⠀⠀⣠⠟⠁⠀⠀⠀⣀⠀⠀⠀⠀⢀⡟⠈⢀⣤⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠙⣏⡁⠀⠐⠚⠃⣿⠀
                ⠀⠀⠀⠀⣴⠋⠀⠀⠀⡴⣿⣿⡟⣷⠀⠀⠊⠀⠴⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠀⠀⠀⠀⢹⡆
                ⠀⠀⠀⣴⠃⠀⠀⠀⠀⣇⣿⣿⣿⠃⠀⠀⠀⠀⠀⠀⠀⠀⢀⣤⡶⢶⣶⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇
                ⠀⠀⣸⠃⠀⠀⠀⢠⠀⠊⠛⠉⠁⠀⠀⠀⠀⠀⠀⠀⢲⣾⣿⡏⣾⣿⣿⣿⣿⠖⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢧
                ⠀⢠⡇⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠈⠛⠿⣽⣿⡿⠏⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡜
                ⢀⡿⠀⠀⠀⠀⢀⣤⣶⣟⣶⣦⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇
                ⢸⠇⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇
                ⣼⠀⢀⡀⠀⠀⢷⣿⣿⣿⣿⣿⣿⡿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⡇
                ⡇⠀⠈⠀⠀⠀⣬⠻⣿⣿⣿⡿⠙⠀⠀⢀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⠁
                ⢹⡀⠀⠀⠀⠈⣿⣶⣿⣿⣝⡛⢳⠭⠍⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⠃⠀
                ⠸⡇⠀⠀⠀⠀⠙⣿⣿⣿⣿⣿⣿⣷⣦⣀⣀⣀⣤⣤⣴⡶⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⠇⠀⠀
                ⠀⢿⡄⠀⠀⠀⠀⠀⠙⣇⠉⠉⠙⠛⠻⠟⠛⠛⠉⠙⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡰⠋⠀⠀⠀
                ⠀⠈⢧⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠞⠁⠀⠀⠀⠀
                ⠀⠀⠘⢷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠞⠁⠀⠀⠀⠀⠀⠀
                ⠀⠀⠀⠀⠱⢆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡴⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀
                ⠀⠀⠀⠀⠀⠀⠛⢦⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣠⠴⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
                ⠀⠀⠀⠀⠀⠀⠀⠀⠈⠛⠲⠤⣤⣤⣤⣄⠀⠀⠀⠀⠀⠀⠀⢠⣤⣤⠤⠴⠒⠛⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀    
    
                                    Feel Lucky?
                                      * * *
                This is not a real ABACUS running_*.log file, instead, 
                this file only works for Phonopy interface calculation.
                                      * * *
'''
    out += f' TOTAL ATOM NUMBER = {len(elem)}\n'
    out += '-'*(22+4+20*3) + '\n'
    out += ' TOTAL-FORCE (eV/Angstrom)\n'
    out += '-'*(22+4+20*3) + '\n'
    
    count = dict.fromkeys(elem, 1) # count the number of atoms for each element from 1
    for e, f in zip(elem, forces):
        temp = str(e) + str(count[e])
        out += f'{temp:>26s}{f[0]:>20.10f}{f[1]:>20.10f}{f[2]:>20.10f}\n'
        count[e] += 1
    out += '-'*(22+4+20*3) + '\n'
    
    out += '''
                                      * * *
                If Phonopy says this is acceptable, then it is acceptable.
                If you are not happy with my doge, you write your own code.
                                Thanks very much!
                                      * * *
'''
    with open(fn, 'w') as f:
        f.write(out)
    return fn

def read_forces(fn):
    '''read the atomic forces from ABACUS running_*.log'''
    logging.info(f'Read forces from {fn} >>')
    
    with open(fn, 'r') as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    lines = [l for l in lines if l] # remove empty lines
    
    # get the number of atoms
    nat, idx = None, []
    for l in lines:
        if l.startswith('TOTAL ATOM NUMBER ='):
            nat = int(l.split()[-1])
            continue
        if l.startswith('TOTAL-FORCE'):
            idx.append(lines.index(l) + 2)
            continue
    if not all([nat, idx]):
        errmsg = f'Cannot find the number of atoms in {fn}'
        logging.error(errmsg)
        raise ValueError(errmsg)
    forces = [l.split() for i in idx for l in lines[i:i+nat]]
    elem = [i[0] for i in forces[:nat]]
    elem = [re.match(r'([A-Z][a-z]?)\d+', i).group(1) for i in elem]
    forces = np.array([i[1:] for i in forces], dtype=float).reshape(-1, nat, 3) # trajectories...
    
    logging.info(f'<< Read forces from {fn}')
    return [(elem, i) for i in forces]

def _split_stru(contents):
    '''split the ABACUS STRU file into sections'''
    delim = ['ATOMIC_SPECIES', 'NUMERICAL_ORBITAL', 'LATTICE_CONSTANT',
             'NUMERICAL_DESCRIPTOR', 'LATTICE_VECTORS', 'ATOMIC_POSITIONS']
    
    ptr = [[None, len(contents)] for _ in delim]
    # find the start of each section
    for i, l in enumerate(contents):
        first = re.split(r'\s+|#', l)[0]
        for j, d in enumerate(delim):
            if first == d:
                ptr[j][0] = i
                break
    
    # find the end of each section
    for i, p in enumerate(ptr): # for each section pointer...
        if p[0] is None: # if the start is not found, do not need to find the end
            continue
        for j, l in enumerate(contents[p[0]+1:]): # start from the next line
            first = re.split(r'\s+|#', l)[0]
            if first in delim: # if the next section is found, break
                ptr[i][1] = p[0] + j
                break

    # now, extract the contents of each section
    out = {}
    for i, p in enumerate(ptr):
        if p[0] is None:
            continue
        out[delim[i]] = [l for l in contents[p[0]+1:p[1]] if l.strip()]
    return out

def _parse_stru_atom(l):
    '''parse the line defining for each atom'''
    pat = r'\d+(\.\d+)?'
    l = [float(i) if re.match(pat, i) else i for i in ' '.join(l).split('#')[0].split()]
    # the first three are always xyz, then the mobility
    n_ld_num = len(l) if all([isinstance(i, float) for i in l]) else [isinstance(i, float) for i in l].index(False)
    if n_ld_num not in [3, 6]:
        errmsg = f'The number of elements in the line is {n_ld_num}, which is not expected: {l}'
        logging.error(errmsg)
        raise ValueError(errmsg)
    
    out = {'xyz': l[:3]}
    for _ in range(3):
        l.pop(0)
    
    if l and n_ld_num == 6:
        out['m'] = l[:3]
        for _ in range(3):
            l.pop(0)
    if l: # the rest will be uniformly in the format of a string leading several numbers
        ik = [i for i, j in enumerate(l) if not isinstance(j, float)]
        # group all adjacent numbers into a list, it will be the value of the leading string
        out.update({l[ik[i]]: [float(j) for j in l[ik[i]+1:ik[i+1]]] for i in range(len(ik)-1)})
        out.update({l[ik[-1]]: [float(j) for j in l[ik[-1]+1:]]})
    if 'm' in out:
        out['m'] = [bool(i) for i in out['m']]
    return out

def _parse_stru(k, v):
    '''parse sections of ABACUS STRU file into dict'''
    if k == 'ATOMIC_POSITIONS': # this is the most complicated
        # the atomic positions are written in the following format:
        # Direct/Cartesian...
        # [element]
        # [magmom]
        # [nat]
        # [x] [y] [z] ...
        i = 0
        while i < len(v):
            if v[i].strip().startswith('Direct') or v[i].strip().startswith('Cartesian'):
                break
            i += 1
        if i == len(v):
            raise ValueError('The coordinate system identifier (e.g. `Direct` or `Cartesian`) are not found')
        out = {'coordinate': v[i].strip()}
        v = ' '.join([l.strip().split('#')[0] for l in v[i+1:]])
        # build the element information
        pat = r'([A-Z][a-z]?)\s+(\d+(\.\d+)?)\s+(\d+)\s+' # replace all this pattern to `@delimiter`
        elem = re.findall(pat, v)
        elem = [dict(zip(['elem', 'magmom', 'nat'], [e[0], float(e[1]), int(e[3])])) for e in elem]
        # then read the atoms
        delim = '@delimiter' + str(uuid.uuid4())
        # split all atoms information into elements
        atoms = [l.strip() for l in re.sub(pat, delim, v).split(delim)]
        # remove empty lines
        atoms = [l for l in atoms if l]
        # reshape with respect to the number of atoms, now atoms will be list of list: [elem][atom] -> data
        atoms = [np.array(l.split()).reshape(e['nat'], -1).tolist() for l, e in zip(atoms, elem)]
        # parse the atoms line by line
        atoms = [[_parse_stru_atom(l) for l in a] for a in atoms]
        # for atoms within the same element group, merge them
        return out|{'atoms': [e|_listdict2dictlist(a) for e, a in zip(elem, atoms)]}
    else:
        return [l.split() for l in v]

def read_stru(fn):
    '''read the ABACUS STRU file (quite laborious)'''

    # split contents with delimeters above, and use one dict to store the
    # split results
    with open(fn, 'r') as f:
        contents = f.readlines()

    out = _split_stru(contents)
    out = {k: _parse_stru(k, v) for k, v in out.items()}
    return out

def write_cell(cell, fn):

    elem = cell.get('elem', ['Fe'])
    psp = cell.get('psp', [f'{elem}.upf'])
    orb = cell.get('orb', [f'{elem}_gga_10au_100Ry_4s2p2d1f.orb'])

    latconst = cell.get('latconst', 1.8897259886)
    latvec = cell.get('latvec', np.eye(3, 3) * latconst)
    atoms = cell.get('atoms', [(0, 0, 0)])

    out = 'ATOMIC_SPECIES\n'
    for e, p in zip(elem, psp):
        out += f'{e} 1.000 {p}\n'
    
    out += '\nNUMERICAL_ORBITAL\n'
    for o in orb:
        out += f'{o}\n'

    out += f'\nLATTICE_CONSTANT\n{latconst}\n'
    
    out += '\nLATTICE_VECTORS\n'
    for vec in latvec:
        out += f' {vec[0]:.10f} {vec[1]:.10f} {vec[2]:.10f}\n'
    out += f'''
ATOMIC_POSITIONS
Direct
'''
    for e, atom_elem in zip(elem, atoms):
        out += f'''
{e}
0.00
{len(atom_elem)}
'''
        for atom in atom_elem:
            out += f'{atom[0]:.10f} {atom[1]:.10f} {atom[2]:.10f} m 1 1 1 mag 0 0 1 sc 1 1 1\n'
    
    with open(fn, 'w') as f:
        f.write(out)
    return fn

def build_supercell(elem, abc, atoms, supercell = 1):
    '''
    generate an ideal cell for the given element and phase, 
    now only support bcc and fcc conventional cells

    Parameters
    ----------
    elem : str
        element symbol
    abc : list
        lattice constants
    atoms : list
        atoms in the unit cell
    supercell : int
        supercell size
    '''
    latvec, atoms = _build_supercell(abc, atoms, supercell)

    cell = {'elem': [elem], 'psp': [f'{elem}.upf'], 'orb': [f'{elem}_gga_10au_100Ry_4s2p2d1f.orb'],
            'latconst': 1.8897259886, 'latvec': latvec, 
            'atoms': [atoms]}

    return cell

def read_energies(flog, unit = "eV", term = "EKS"):
    '''read the energy terms from the ABACUS output file
    This implementation is copied from:
    https://github.com/kirk0830/ABACUS-Pseudopot-Nao-Square/blob/58d4693b9e9c7567c42907bdd8727f59533c8021/apns/analysis/postprocess/read_abacus_out.py#L338
    '''
    harris = ["Harris", "harris", "HARRIS", "eharris", "EHARRIS", "eh"]
    fermi = ["Fermi", "fermi", "FERMI", "efermi", "EFERMI", "ef"]
    kohnsham = ["KohnSham", "kohnsham", "KOHN", "kohn", "KOHNSHAM", 
                "ekohnsham", "EKOHN", "EKOHNSHAM", "eks", "EKS", "e", "E", "energy"]
    if term in harris:
        header = "E_Harris"
    elif term in fermi:
        header = "E_Fermi"
    elif term in kohnsham:
        header = "E_KohnSham"
    else:
        raise ValueError("Unknown energy term")
    with open(flog, "r") as f:
        eners = [float(line.split()[-1]) for line in f.readlines() if line.strip().startswith(header)]
    eners = [convert_energy_unit(ener, "eV", unit) for ener in eners]
    return np.array(eners, dtype=np.float64)

def read_natoms(flog):
    '''read the total number of atoms from the ABACUS output file
    This implementation is copied from: https://github.com/kirk0830/ABACUS-Pseudopot-Nao-Square/blob/58d4693b9e9c7567c42907bdd8727f59533c8021/apns/analysis/postprocess/read_abacus_out.py#L362
    '''
    with open(flog, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    for line in lines:
        if line.startswith("TOTAL ATOM NUMBER"):
            return int(line.split()[-1])

def read_final_energy(flog, unit = "eV"):
    '''read the final energy from the ABACUS output file.
    we can quickly get it by `grep "!" [flog]` in the command line
    '''
    ftmp = flog + ".tmp"
    os.system(f'grep "@_@" {flog} > {ftmp}') 
    # the signal that SCF convergence is not reached
    with open(ftmp, "r") as f:
        content = f.readlines()
    content = [line.strip() for line in content]
    content = [line for line in content if line]
    # if not empty, return None
    if content:
        return None
    # if empty, read the final energy
    os.system(f'grep "!" {flog} > {ftmp}')
    with open(ftmp, "r") as f:
        line = f.readlines()[0]
    # will be something like '!FINAL_ETOT_IS -13721.7598374452409189 eV'
    energy = float(line.split()[-2])
    os.remove(ftmp)
    return convert_energy_unit(energy, "eV", unit)

def isfinished(root):
    '''check if the calculation is finished in one folder'''
    fdft = os.path.join(root, 'INPUT')
    try:
        # if any of the following lines raise FileNotFoundError, the calculation is not finished
        dft = read_fdft(fdft)
        suffix = dft.get('suffix', 'ABACUS')
        cal_type = dft.get('calculation', 'scf')
        flog = os.path.join(root, f'OUT.{suffix}', f'running_{cal_type}.log')
        # if successfully arrive here, read flog. ABACUS now will end with:
        # Start  Time  : ***
        # Finish Time  : ***
        # Total  Time  : ***
        # check them
        with open(flog, 'r') as f:
            lines = f.readlines()
        lines = [l.strip() for l in lines]
        lines = [l for l in lines if l]
        if not lines:
            return False
        if not re.match(r'^Start\s+Time\s*:', lines[-3]):
            return False
        if not re.match(r'^Finish\s+Time\s*:', lines[-2]):
            return False
        if not re.match(r'^Total\s+Time\s*:', lines[-1]):
            return False
        return True
    except FileNotFoundError:
        return False

class TestDeltaSpinABACUSUtils(unittest.TestCase):
    '''test the utility functions for ABACUS'''
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

    def test_write_cp2k_forces(self):
        '''test _write_cp2k_forces'''
        fn = 'test-forces.xyz'
        elem = ['Fe', 'Fe']
        forces = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        _ = _write_cp2k_forces(fn, elem, forces)
        self.assertTrue(os.path.exists(fn))
        os.remove(fn)

    def test_parse_stru_atom(self):
        '''test _parse_stru_atom'''
        l = ['0.0000000000', '0.0000000000', '0.0000000000', 'm', '1', '1', '1', 'mag', '0', '0', '2.40', 'sc', '1', '1', '1']
        out = _parse_stru_atom(l)
        self.assertTrue('xyz' in out)
        self.assertEqual(out['xyz'], [0.0, 0.0, 0.0])
        self.assertTrue('m' in out)
        self.assertEqual(out['m'], [True, True, True])
        self.assertTrue('mag' in out)
        self.assertEqual(out['mag'], [0.0, 0.0, 2.4])
        self.assertTrue('sc' in out)
        self.assertEqual(out['sc'], [1.0, 1.0, 1.0])
        
        l = '0.5 0.0 0.5 1 1 1 vel 0.0 0.0 0.0 angle1 0.0 angle2 60.0 # this is a comment'.split()
        out = _parse_stru_atom(l)
        self.assertTrue('xyz' in out)
        self.assertEqual(out['xyz'], [0.5, 0.0, 0.5])
        self.assertTrue('vel' in out)
        self.assertEqual(out['vel'], [0.0, 0.0, 0.0])
        self.assertTrue('angle1' in out)
        self.assertEqual(out['angle1'], [0.0])
        self.assertTrue('angle2' in out)
        self.assertEqual(out['angle2'], [60.0])
        
        l = '0 0 0'.split()
        out = _parse_stru_atom(l)
        self.assertTrue('xyz' in out)
        self.assertEqual(out['xyz'], [0.0, 0.0, 0.0])
        
        l = '0 0 0 m 1 1 1'.split()
        out = _parse_stru_atom(l)
        self.assertTrue('xyz' in out)
        self.assertEqual(out['xyz'], [0.0, 0.0, 0.0])
        self.assertTrue('m' in out)
        self.assertEqual(out['m'], [True, True, True])
        
        l = '0 0 0 1 0 1'.split()
        out = _parse_stru_atom(l)
        self.assertTrue('xyz' in out)
        self.assertEqual(out['xyz'], [0.0, 0.0, 0.0])
        self.assertTrue('m' in out)
        self.assertEqual(out['m'], [True, False, True])

    def test_parse_stru(self):
        src = '''Direct
Fe # Hello, I am a comment
0.00
4
0.0000000000 0.0000000000 0.0000000000
0.5000000000 0.5000000000 0.0000000000
0.5000000000 0.0000000000 0.5000000000
0.0000000000 0.5000000000 0.5000000000
# should ignored
O
0.00
2
0.2500000000 0.2500000000 0.2500000000 m 1 1 1
0.7500000000 0.7500000000 0.7500000000 m 1 1 1

S
15.123456
1
0.8000000000 0.8000000000 0.9000000000 m 1 1 1 mag 0 0 2.40 sc 1 1 1

'''
        k = 'ATOMIC_POSITIONS'
        v = src.split('\n')
        out = _parse_stru(k, v)
        self.assertTrue('coordinate' in out)
        self.assertEqual(out['coordinate'], 'Direct')
        self.assertTrue('atoms' in out)
        self.assertEqual(len(out['atoms']), 3) # 3 elements
        self.assertTrue('Fe' in [i['elem'] for i in out['atoms']])
        self.assertTrue('O' in [i['elem'] for i in out['atoms']])
        self.assertTrue('S' in [i['elem'] for i in out['atoms']])
        
    def test_split_stru(self):
        fn = self.init_mytest()
        with open(fn, 'r') as f:
            contents = f.readlines()
        os.remove(fn)
        out = _split_stru(contents)
        self.assertTrue('ATOMIC_SPECIES' in out)
        self.assertEqual(out['ATOMIC_SPECIES'], ['Fe 55.845 Fe.pbe-spn-rrkjus_psl.1.0.0.UPF\n'])
        self.assertTrue('NUMERICAL_ORBITAL' in out)
        self.assertEqual(out['NUMERICAL_ORBITAL'], ['Fe_gga_10au_100Ry_4s2p2d1f.orb\n'])
        self.assertTrue('LATTICE_CONSTANT' in out)
        self.assertEqual(out['LATTICE_CONSTANT'], ['1.889726125457828\n'])
        self.assertTrue('LATTICE_VECTORS' in out)
        self.assertEqual(out['LATTICE_VECTORS'], ['3.405 0.000 0.000\n', '0.000 3.405 0.000\n', '0.000 0.000 3.405\n'])
        self.assertTrue('ATOMIC_POSITIONS' in out)
        self.assertEqual(out['ATOMIC_POSITIONS'], 
                         ['Direct\n', 'Fe\n', '0.00\n', '2\n', '0.00 0.00 0.00 m 1 1 1 mag 0 0 2.40 sc 1 1 1\n', '0.50 0.50 0.50 m 1 1 1 mag 0 0 2.40 sc 1 1 1\n'])
        
    def test_read_stru(self):
        fn = self.init_mytest()
        out = read_stru(fn)
        os.remove(fn)
        self.assertTrue('ATOMIC_SPECIES' in out)
        self.assertTrue('NUMERICAL_ORBITAL' in out)
        self.assertTrue('LATTICE_CONSTANT' in out)
        self.assertTrue('LATTICE_VECTORS' in out)
        self.assertTrue('ATOMIC_POSITIONS' in out)
        self.assertEqual(out['ATOMIC_POSITIONS']['coordinate'], 'Direct')
        self.assertTrue('atoms' in out['ATOMIC_POSITIONS'])
        self.assertEqual(len(out['ATOMIC_POSITIONS']['atoms']), 1)

    def testwrite_forces(self):
        '''test write_forces'''
        fn = 'test-forces.log'
        elem = ['Fe'] * 32
        forces = [0.0000000000, 0.0000000000, 0.0000000000,
                  0.5000000000, 0.0000000000, 0.0000000000,
                  0.0000000000, 0.5000000000, 0.0000000000,
                  0.5000000000, 0.5000000000, 0.0000000000,
                  0.0000000000, 0.0000000000, 0.5000000000,
                  0.5000000000, 0.0000000000, 0.5000000000,
                  0.0000000000, 0.5000000000, 0.5000000000,
                  0.5000000000, 0.5000000000, 0.5000000000,
                  0.2500000000, 0.2500000000, 0.0000000000,
                  0.7500000000, 0.2500000000, 0.0000000000,
                  0.2500000000, 0.7500000000, 0.0000000000,
                  0.7500000000, 0.7500000000, 0.0000000000,
                  0.2500000000, 0.2500000000, 0.5000000000,
                  0.7500000000, 0.2500000000, 0.5000000000,
                  0.2500000000, 0.7500000000, 0.5000000000,
                  0.7500000000, 0.7500000000, 0.5000000000,
                  0.2500000000, 0.0000000000, 0.2500000000,
                  0.7500000000, 0.0000000000, 0.2500000000,
                  0.2500000000, 0.5000000000, 0.2500000000,
                  0.7500000000, 0.5000000000, 0.2500000000,
                  0.2500000000, 0.0000000000, 0.7500000000,
                  0.7500000000, 0.0000000000, 0.7500000000,
                  0.2500000000, 0.5000000000, 0.7500000000,
                  0.7500000000, 0.5000000000, 0.7500000000,
                  0.0000000000, 0.2500000000, 0.2500000000,
                  0.5000000000, 0.2500000000, 0.2500000000,
                  0.0000000000, 0.7500000000, 0.2500000000,
                  0.5000000000, 0.7500000000, 0.2500000000,
                  0.0000000000, 0.2500000000, 0.7500000000,
                  0.5000000000, 0.2500000000, 0.7500000000,
                  0.0000000000, 0.7500000000, 0.7500000000,
                  0.5000000000, 0.7500000000, 0.7500000000]
        forces = np.array(forces).reshape(-1, 3)
        _ = write_forces(fn, elem, forces)
        self.assertTrue(os.path.exists(fn))
        os.remove(fn)        
    
    @unittest.skip('Due to the refactor of code, the file to read does not exist')
    def test_read_forces(self):
        '''test read_forces'''
        fn = 'testfiles/running_cell-relax.log'
        out = read_forces(fn)
        # there is only one frame in this file
        self.assertEqual(len(out), 1)
        elem, forces = out[0]
        self.assertEqual(len(elem), 2)
        self.assertEqual(len(set(elem)), 1)
        self.assertEqual(elem[0], 'Fe')
        self.assertEqual(forces.shape, (2, 3))

    def test_build_supercell(self):
        cell = build_supercell('Fe', [2.86304]*3, [[0, 0, 0], [0.5, 0.5, 0.5]], 4)
        self.assertEqual(len(cell['atoms'][0]), 4**3 * 2)
        self.assertEqual(cell['latvec'].shape, (3, 3))

    def test_read_final_energy(self):
        '''test read_final_energy'''
        fn = 'testfiles/running_scf.log'
        e = read_final_energy(fn)
        self.assertAlmostEqual(e, -13721.7598374452409189)

    def test_is_finished(self):
        '''test isfinished'''
        # we will temporarily do file I/O in the testfiles folder
        root = 'testfiles'
        
        # first build a successful case
        folder = f'temp-finished'
        folder = os.path.join(root, folder)
        os.makedirs(folder, exist_ok=True)
        fdft = os.path.join(folder, 'INPUT')
        with open(fdft, 'w') as f:
            f.write('INPUT_PARAMETERS\ncalculation scf\nsuffix DeltaSpinXTest\n')
        outdir = os.path.join(folder, 'OUT.DeltaSpinXTest')
        os.makedirs(outdir, exist_ok=True)
        flog = os.path.join(outdir, 'running_scf.log')
        with open(flog, 'w') as f:
            f.write('Start Time: 2021-09-09 12:00:00\nFinish Time: 2021-09-09 12:00:01\nTotal Time: 1s\n')
        self.assertTrue(isfinished(folder))
        shutil.rmtree(folder)
        
        # then build failed cases
        # case 1: no such a ABACUS job root
        folder = f'temp-incomplete'
        folder = os.path.join(root, folder)
        self.assertFalse(isfinished(folder))
        # case 2: no ABACUS job INPUT
        os.makedirs(folder, exist_ok=True)
        self.assertFalse(isfinished(folder))
        # case 3: no ABACUS job outdir
        fdft = os.path.join(folder, 'INPUT')
        with open(fdft, 'w') as f:
            f.write('INPUT_PARAMETERS\ncalculation scf\nsuffix DeltaSpinXTest\n')
        self.assertFalse(isfinished(folder))
        # case 4: no ABACUS job flog
        outdir = os.path.join(folder, 'OUT.DeltaSpinXTest')
        os.makedirs(outdir, exist_ok=True)
        self.assertFalse(isfinished(folder))
        # case 5: flog is not finished
        flog = os.path.join(outdir, 'running_scf.log')
        with open(flog, 'w') as f:
            f.write('Write something not relevant\n')
        self.assertFalse(isfinished(folder))
        shutil.rmtree(folder)

if __name__ == '__main__':
    test = init()
    unittest.main(exit=test)
    # write_cell(build_supercell('Fe', [2.86304]*3, [[0, 0, 0], [0.5, 0.5, 0.5]], 3), 'BCC-Fe-333')
    