# in-built modules
import itertools as it
import re
import unittest

# external modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# home-made modules
from utils import init

def pbcdist(cell, this, other, delta = 1e-6):
    '''calculate the distance between two positions in the periodic boundary condition
    
    Parameters
    ----------
    cell : np.array
        the cell vectors
    this : np.array
        the position of the first atom
    other : np.array
        the position of the other atom or atoms
        
    Returns
    -------
    dist : float
        the distance between the two atoms
    '''
    this = np.array(this)
    if this.shape != (3,):
        raise ValueError(f'this should be a 3-element list or a 3x1 matrix: {this}')
    
    other = np.array(other)
    if other.shape == (3,):
        other = other.reshape(1, 3)
    elif other.shape[1] != 3:
        raise ValueError(f'other should be a 3-element list or a Nx3 matrix: {other}')
    
    disp_d = other - this
    disp_d = (disp_d + 0.5)%1 - 0.5 # in shape of (N, 3)
    if disp_d.shape != other.shape:
        raise RuntimeError(f'disp_d should have the same shape as other: {disp_d.shape} != {other.shape}')
    
    # get N distances in Cartesian coordinates
    dist_c = np.linalg.norm(disp_d @ cell, axis = 1)
    # reduce to given delta
    return np.round(dist_c, int(-np.log10(delta)))

def abc_angles_to_vec(lat: list, as_list: bool = False):
    """convert lattice parameters to vectors
    
    copied from https://github.com/kirk0830/ABACUS-Pseudopot-Nao-Square/blob/58d4693b9e9c7567c42907bdd8727f59533c8021/apns/test/atom_species_and_cell.py#L378"""
    assert len(lat) == 6, f'lat should be a list of 6 floats: {lat}'
    a, b, c, alpha, beta, gamma = lat
    alpha, beta, gamma = np.deg2rad(alpha), np.deg2rad(beta), np.deg2rad(gamma)
    e11, e12, e13 = a, 0, 0
    e21, e22, e23 = b * np.cos(gamma), b * np.sin(gamma), 0
    e31 = c * np.cos(beta)
    e32 = (b * c * np.cos(alpha) - e21 * e31) / e22
    e33 = np.sqrt(c**2 - e31**2 - e32**2)
    return np.array([[e11, e12, e13], [e21, e22, e23], [e31, e32, e33]])\
        if not as_list else [[e11, e12, e13], [e21, e22, e23], [e31, e32, e33]]

def vec_to_abc_angles(cell: np.array, deg: bool = True):
    """convert lattice vectors to lattice parameters
    
    """
    a = np.linalg.norm(cell[0])
    b = np.linalg.norm(cell[1])
    c = np.linalg.norm(cell[2])
    alpha = np.arccos(np.dot(cell[1], cell[2]) / b / c)
    beta = np.arccos(np.dot(cell[0], cell[2]) / a / c)
    gamma = np.arccos(np.dot(cell[0], cell[1]) / a / b)
    
    if deg:
        alpha = np.rad2deg(alpha)
        beta = np.rad2deg(beta)
        gamma = np.rad2deg(gamma)
    return a, b, c, alpha, beta, gamma

def _special_qpoints(latname, alat, direct = True):
    '''generate q points in the first Brillouin zone.
    
    This function has been tested by SeeK-path package

    Parameters
    ----------
    latname : str
        lattice name: 'bcc', 'fcc', 'hex'
    alat : float
        lattice constant of the conventional cell, in Angstrom
    direct : bool
        whether to return the q points in direct coordinates. 
        If False, return in Cartesian coordinates, unit in 1/Angstrom,
        otherwise, return the one based on the reciprocal lattice vectors.
    '''
    from ase.lattice import BCC, FCC, HEX
    constructor = {'bcc': BCC, 'fcc': FCC, 'hex': HEX}
    alat = [alat] if not isinstance(alat, list) else alat
    crystal = constructor[latname](*alat) # specified by the conventional cell dimension

    cell = np.array(crystal.tocell())
    
    # the special points in the first Brillouin zone
    special = crystal.get_special_points()

    out = special if direct else {k: np.dot(v, np.linalg.inv(cell).T) * 2 * np.pi \
                                  for k, v in special.items()}
    return out

def qpathgen(latname, alat, path = None, direct = True, n = 10):
    '''generate q points' path for the first Brillouin zone
    
    Parameters
    ----------
    latname : str
        lattice name: 'bcc', 'fcc', 'hex'
    alat0 : float
        lattice constant of the conventional cell, in Angstrom
    path : str
        path name, e.g. 'GNPGHN' for bcc
    n : int
        number of points between two special points
    
    Returns
    -------
    selected : list
        selected special points
    path : np.array
        q points' path, unit in 2pi/Angstrom
    '''
    special = _special_qpoints(latname, alat, direct=direct)
    selected = list(special.keys()) if path is None \
        else [p for p in re.findall(r'([A-Z])', path)]
    selected = [special[q] for q in selected]
    
    path = []
    for i in range(len(selected) - 1):
        q1 = selected[i]
        q2 = selected[i + 1]
        path.extend([q1 + (q2 - q1) * j / n for j in range(n)])
    path.append(q2)
    
    return selected, np.array(path)

def _build_supercell(abc, atoms, n):
    '''
    build a supercell of the crystal lattice
    
    Parameters
    ----------
    abc : list
        lattice constant of the conventional cell, in Angstrom
    atoms : np.array
        fractional coordinates of atoms in the conventional cell
    n : int
        the number of duplications of the original cell
        
    Returns
    -------
    cell : np.array
        the cell vectors of the supercell
    atoms : np.array
        the positions of atoms in the supercell in fractional coordinates
    '''
    abc = np.array(abc)
    cell = np.diag(abc) if abc.shape == (3,) else abc
    if cell.shape != (3, 3):
        raise ValueError(f'abc should be a 3-element list or a 3x3 matrix: {cell}')
    if np.linalg.det(cell) == 0:
        raise ValueError(f'The cell defined is singular: {cell}')
    
    atoms = np.array(atoms)
    atoms = [(atom + np.array([i, j, k]))/n \
             for i, j, k in it.product(range(n), repeat = 3) \
             for atom in atoms]
    atoms = np.array(atoms).reshape(-1, 3)
    atoms = sorted(atoms, key = lambda x: np.linalg.norm(x))
    
    return cell * n, atoms

def supercell_count(abc, nshell, atoms, mode='reach'):
    '''
    calculate the number of supercell needed to reach the `nshell` shells
    or to fully include the `nshell` shells in the crystal lattice
    
    Parameters
    ----------
    abc : list
        lattice constant of the conventional cell, in Angstrom
    nshell : int
        number of shells to consider
    atoms : np.array
        fractional coordinates of atoms in the conventional cell
    mode : str
        whether to reach the `nshell` shells or to fully include the `nshell` shells
    
    Return
    ------
    n : int
        the number of duplications of the original cell, or say the number of additional
        shells introduced by the supercell. If not needed to introduce any supercell, then
        return 0.
    '''
    if nshell <= 0:
        raise ValueError(f'nshell should be a positive integer: {nshell}')
    
    abc = np.array(abc)
    cell = np.diag(abc) if abc.shape == (3,) else abc
    if cell.shape != (3, 3):
        raise ValueError(f'abc should be a 3-element list or a 3x3 matrix: {cell}')
    if np.linalg.det(cell) == 0:
        raise ValueError(f'The cell defined is singular: {cell}')
    
    if mode not in ['reach', 'full']:
        raise ValueError(f'mode should be either "reach" or "full": {mode}')
    
    atoms = np.array(atoms)        
    n = 0
    # increase the number of supercells till i == nshell to reach the convergence
    # of the number of atoms in each shell in the range of interest
    
    # the "spherical cell limit": only when it is spherical, then introducing
    # one supercell will exactly introduce one shell, otherwise there will be
    # more than one shell introduced. So nshell of supercell will at least
    # introduce nshell shells
    if mode == 'reach': 
        while n <= 2*nshell + 1:
            tau_d = np.array([np.array([i, j, k] + atom) / (n + 1)
                             for i, j, k in it.product(range(n + 1), repeat = 3)
                             for atom in atoms])
            origin = tau_d[0]
            dist = pbcdist(cell*(n + 1), this=origin, other=tau_d, delta=1e-6)
            dist, count = np.unique(dist, return_counts=True)
            if len(count) >= nshell + 1: # once there are atoms in the nshell-th shell
                break
            n += 1 # otherwise increase the number of supercells
        return n
    else:
        # unlike the 'reach' mode, the 'full' mode requires the number of atoms in
        # shells of interest converges towards any further addition of supercells
        count = np.array([0] * (nshell + 1))
        while n <= 2*nshell + 1:
            tau_d = np.array([np.array([i, j, k] + atom) / (n + 1)
                             for i, j, k in it.product(range(n + 1), repeat = 3)
                             for atom in atoms])
            origin = tau_d[0]
            dist = pbcdist(cell*(n + 1), this=origin, other=tau_d, delta=1e-6)
            dist, c = np.unique(dist, return_counts=True)
            c.resize(nshell + 1)
            if np.all(c == count): # once the number of atoms in each shell converges
                break
            count = c # otherwise update the count, and increase the number of supercells
            n += 1
        return n - 1 # because we can only know whether the count converges after the last addition
                              
def _radial_count(abc, nshell, atoms, direct = True, delta = 1e-6):
    '''
    count the distribution (number) of atoms in the radial direction

    Algorithm
    ---------
    For example, in bcc conventional lattice, there are two atoms: (0, 0, 0) 
    and (0.5, 0.5, 0.5). we duplicate the whole lattice with nshell times in 
    (+/-x, +/-y, +/-z) and calculate the distance between the origin 
    (0, 0, 0) and the duplicated atoms.
    
    Parameters
    ----------
    abc : list
        lattice constant of the conventional cell, in Angstrom
    nshell : int
        number of shells to consider
    atoms : np.array
        positions of atoms in the conventional cell
    direct : bool
        whether to return the direct coordinates of atoms
    
    Returns
    -------
    dist : np.array
        the radial distance
    count : np.array
        the number of atoms in each shell
    tau : np.array
        the positions of atoms in the radial direction
    '''
    abc = np.array(abc)
    cell = np.diag(abc) if abc.shape == (3,) else abc
    if cell.shape != (3, 3):
        raise ValueError(f'abc should be a 3-element list or a 3x3 matrix: {cell}')
    atoms = np.array(atoms)

    ncell = nshell # each cell will introduce at least one atomic shell

    # direct coordinates
    tau_d = np.array([np.array([i, j, k] + atom)
                      for i, j, k in it.product(range(-ncell, ncell+1), repeat = 3)
                      for atom in atoms])
    # cartesian coordinates
    tau_c = tau_d @ cell 
    # sort by distance to the origin
    idx = np.argsort(np.linalg.norm(tau_c, axis = 1))
    tau_c = tau_c[idx]
    # calculate unique distances and their counts
    dist, count = np.unique(np.round(np.linalg.norm(tau_c, axis = 1), 
                                     decimals=int(-np.log10(delta))), 
                            return_counts = True)
    
    # remove the origin and cut the first nshell shells
    dist = dist[1:nshell+1]
    count = count[1:nshell+1]
    tau = tau_c if not direct else tau_d[idx]
    tau = tau[1:np.sum(count)+1]
    
    return dist, count, tau

def rdf(abc, nshell, atoms, dr = 0.01, smear = None):
    '''
    calculate the radial distribution function for the crystal lattice
    
    The g(r) has such a definition:
    g(r) = 1 / (4 * pi * r^2) * dN(r) / dr
    , which means in an interval [r, r + dr], the number of atoms
    is dN(r) = g(r) * 4 * pi * r^2 * dr

    Parameters
    ----------
    abc : list
        lattice constant of the conventional cell, in Angstrom
    nshell : int
        number of shells to consider
    atoms : np.array
        positions of atoms in the conventional cell
    smear : float or None
        the width of the Gaussian smearing function, in Angstrom

    Returns
    -------
    radial : np.array
        the radial distance
    out : np.array
        the radial distribution function
    
    Notes
    -----
    the correctness of this function has been verified by the paper
    Haeberle J, Sperl M, Born P. 
    Distinguishing noisy crystalline structures using bond orientational order parameters[J]. 
    The European Physical Journal E, 2019, 42: 1-7.

    see unittest of this function for usage and other details
    '''
    dist, count, _ = _radial_count(abc, nshell, atoms, direct=False)
    # r can be determined by the max(dist) and dr
    radial = np.arange(0, np.max(dist * 1.1), dr) # we add 10% more space
    # zero padding to get g(r)
    out = np.zeros_like(radial)

    i = 0
    for j, r in enumerate(radial):
        if r < dist[0]:
            out[j] = 0
            continue
        while i < len(dist):
            if np.abs(dist[i] - r) < dr:
                out[j] = count[i] / (4 * np.pi * r**2 * dr)
                i += 1
            else:
                break
    # smooth the g(r) by a Gaussian smearing function
    out = gaussian_filter1d(out, smear / dr) if smear is not None else out
    return radial, out

def neighborgen(latname, alat, nshell, direct = True, group = False):
    '''
    return the un-smoothed, un-normalized radial distribution function
    for the crystal lattice.

    Parameters
    ----------
    latname : str
        lattice name: 'bcc', 'fcc', 'hex'. The capital letter means
        the use of the conventional cell, while the lower case means
        the primitive cell.
    alat : float
        lattice constant of the conventional cell, in Angstrom
    nshell : int
        number of shells/peaks of rdf to consider
    direct : bool
        whether to return the direct coordinates of atoms
    group : bool
        whether to group the position of atoms in each shell

    Returns
    -------
    dist : np.array
        the radial distance
    count : np.array
        the number of atoms in each shell
    tau : np.array
        the positions of atoms in the radial direction. If `group`
        is True, then it is a list of np.array, each of which contains
        the positions of atoms in a shell.

    Warning
    -------
    Capital letter of latname means the use of the conventional cell,
    while the lower case means the primitive cell. This will affect the
    returned positions of atoms if `direct` is True.
    '''
    CELLS = {'bcc': 0.5 * np.array([[-alat,  alat,  alat], 
                                    [ alat, -alat,  alat], 
                                    [ alat,  alat, -alat]]),
             'BCC': np.array([[alat, 0, 0],
                              [0, alat, 0],
                              [0, 0, alat]]),
             'fcc': 0.5 * np.array([[0, alat, alat],
                                    [alat, 0, alat],
                                    [alat, alat, 0]]),
             'FCC': np.array([[alat, 0, 0],
                              [0, alat, 0],
                              [0, 0, alat]])}
    ATOMS = {'bcc': np.array([[0, 0, 0]]),
             'BCC': np.array([[0, 0, 0], [0.5, 0.5, 0.5]]),
             'fcc': np.array([[0, 0, 0]]),
             'FCC': np.array([[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])}
    dist, count, tau = _radial_count(CELLS[latname], nshell, ATOMS[latname], direct)
    tau = np.split(tau, np.cumsum(count)[:-1]) if group else tau
    return dist, count, tau

def make_shell_spikes(abacus_stru, nshell, iat, elem=None, return_global=False):
    '''given the `iat`, index the first atoms in the `nshell` shells,
    where the `shell` is defined by the distance to the `iat`.
    
    Parameters
    ----------
    abacus : dict
        the structure data from Abacus
    nshell : int
        the number of shells
    iat : int|tuple[str, int]
        specify the Origin atom, can be either a global index or
        a tuple of element and local (within the type) index.
    elem : str|list|None
        the element of the atoms to count, if None, all atoms
        are counted.
        
    Returns
    -------
    idx : list[tuple[str, int]]
        the element and local index of the first atom of each shell
    '''
    atoms = abacus_stru['ATOMIC_POSITIONS']['atoms'] # allow this to raise error
    elem = [atom['elem'] for atom in atoms] if elem is None else elem
    elem = [elem] if isinstance(elem, str) else elem

    # the origin: (x, y, z)
    elem_unique = [atom['elem'] for atom in atoms]
    pos = np.array([pos for atom in atoms for pos in atom['xyz']]).reshape(-1, 3)
    origin = pos[iat] if isinstance(iat, int) else atoms[elem_unique.index(iat[0])]['xyz'][iat[1]]
    origin = np.array(origin)
    
    # only consider those atoms whose element is in `elem`
    symb = [[atom['elem']] * atom['nat'] for atom in atoms]
    symb = [s for grp in symb for s in grp]
    glb_idx0 = [i for i, e in enumerate(symb) if e in elem]
    dist = np.linalg.norm(pos[glb_idx0] - origin, axis=1)
    loc_idx = np.argsort(dist) # get the index of the sorted distance
    glb_idx = [glb_idx0[i] for i in loc_idx] # global index sorted by distance
    dist_unique, count = np.unique(dist, return_counts=True)

    if dist_unique[0] == 0: # exclude itself if origin is included
        dist_unique = dist_unique[1:]
        count = count[1:]
        glb_idx = glb_idx[1:]
    if len(count) < nshell:
        raise ValueError(f'The number of atoms in the shells is less than request: {len(count)} < {nshell}')
    # for each shell, only get the first atom
    nidx = [0] + np.cumsum(count).tolist()
    idx = [glb_idx[nidx[i]] for i in range(nshell)]
    if return_global:
        return idx
    glbmap = [(atom['elem'], i) for atom in atoms for i in range(atom['nat'])]
    return [glbmap[i] for i in idx]
        
class TestStructureUtils(unittest.TestCase):
    
    def test_radial_count(self):
        # test with one conventional cell of bcc
        out = _radial_count([4, 4, 4], 1, [[0, 0, 0], [0.5, 0.5, 0.5]], False)
        dist, count, _ = out
        self.assertAlmostEqual(dist[0], np.sqrt(3)/2 * 4, delta=1e-5)
        self.assertEqual(count[0], 8)

    def test_neighborgen(self):
        # test primitive cell of bcc
        out = neighborgen('bcc', 2.86304, 4, group=True, direct=False)
        dist, count, pos = out
        self.assertAlmostEqual(dist[-1], np.sqrt(11)/2 * 2.86304, delta=1e-5)
        for c, p in zip(count, pos):
            self.assertEqual(c, len(p))

        # test conventional cell of bcc
        out2 = neighborgen('BCC', 2.86304, 4, group=True, direct=False)
        dist2, count2, pos2 = out2

        self.assertTrue(np.allclose(dist, dist2))
        self.assertTrue(np.allclose(count, count2))
        for i in range(4):
            self.assertTrue(len(pos[i]) == len(pos2[i]))
            self.assertTrue(all([p1 in pos2[i] for p1 in pos[i]]))

    @unittest.skip('The calculation of RDF is not a typical unittest')
    def test_rdf(self):
        dist, g = rdf([2.86304]*3, 10, [[0, 0, 0], [0.5, 0.5, 0.5]], 0.1)
        plt.plot(dist, g, 'r-', linewidth = 2)
        plt.xlabel('r (Angstrom)')
        plt.ylabel('g(r)')
        plt.tight_layout()
        plt.grid(True)
        plt.savefig('TestFig7b-CaseTestRDF.png')

    def test_qpoints(self):
        # referece data from SeeK-path package
        q = _special_qpoints('bcc', [2.86304], direct=True)
        G_direct = [0.0000000000,  0.0000000000, 0.0000000000]
        H_direct = [0.5000000000, -0.5000000000, 0.5000000000]
        N_direct = [0.0000000000,  0.0000000000, 0.5000000000]
        P_direct = [0.2500000000,  0.2500000000, 0.2500000000]

        self.assertTrue(np.allclose(q['G'], G_direct))
        self.assertTrue(np.allclose(q['H'], H_direct))
        self.assertTrue(np.allclose(q['N'], N_direct))
        self.assertTrue(np.allclose(q['P'], P_direct))

        q = _special_qpoints('bcc', [2.86304], direct=False)
        G_cart = [0.0000000000, 0.0000000000, 0.0000000000]
        H_cart = [0.0000000000, 2.1945886829, 0.0000000000]
        N_cart = [1.0972943415, 1.0972943415, 0.0000000000]
        P_cart = [1.0972943415, 1.0972943415, 1.0972943415]

        self.assertTrue(np.allclose(q['G'], G_cart))
        self.assertTrue(np.allclose(q['H'], H_cart))
        self.assertTrue(np.allclose(q['N'], N_cart))
        self.assertTrue(np.allclose(q['P'], P_cart))

    def test_qpathgen(self):
        # test with bcc lattice
        selected, path = qpathgen('bcc', [2.86304], 'GNPGHN', direct=True, n=10)

        special = [np.array([0, 0, 0]), np.array([0. , 0. , 0.5]), 
                   np.array([0.25, 0.25, 0.25]), np.array([0, 0, 0]), 
                   np.array([ 0.5, -0.5,  0.5]), np.array([0. , 0. , 0.5])]
        for s, p in zip(selected, special):
            self.assertTrue(np.allclose(s, p))
        
        self.assertEqual(path.shape, (51, 3))
        
    def test_make_shell_spikes(self):
        abacus = {'ATOMIC_POSITIONS': {'atoms': [
            {'elem': 'Fe', 'nat': 2, 'xyz': [[0, 0, 0], [0.5, 0.5, 0.5]]},
            {'elem': 'Ni', 'nat': 1, 'xyz': [[0.25, 0.25, 0.25]]}
        ]}}
        # if only consider Fe, then the second shell would be ('Fe', 1)
        # if consider all atoms, then the second shell would be ('Ni', 0)
        idx = make_shell_spikes(abacus, 1, 0, elem=['Fe'])
        self.assertEqual(idx, [('Fe', 1)])
        idx = make_shell_spikes(abacus, 1, 0, elem=None)
        self.assertEqual(idx, [('Ni', 0)])
        idx = make_shell_spikes(abacus, 2, 0, elem=None)
        self.assertEqual(idx, [('Ni', 0), ('Fe', 1)])
        idx = make_shell_spikes(abacus, 2, 0, elem=None, return_global=True)
        self.assertEqual(idx, [2, 1])
    
    def test_supercell_count_reach(self):
        # test with one conventional cell of sc
        options = {'abc': [4, 4, 4], 'nshell': 1, 'atoms': [[0, 0, 0]], 'mode': 'reach'}
        # if need to fully include the 1st n.n., then 1 supercell is enough
        n = supercell_count(**options)
        self.assertEqual(n, 1)
        # for 2nd n.n., 1 supercell is also enough
        n = supercell_count(**(options | {'nshell': 2}))
        self.assertEqual(n, 1)
        # for 3rd n.n., 1 supercells are also needed
        n = supercell_count(**(options | {'nshell': 3}))
        self.assertEqual(n, 1)
        # for 4th n.n., 2 supercells are needed
        n = supercell_count(**(options | {'nshell': 4}))
        self.assertEqual(n, 3)
        
        # test with one conventional cell of bcc
        options = {'abc': [2.86304]*3, 'nshell': 1, 'atoms': [[0, 0, 0], [0.5, 0.5, 0.5]], 
                   'mode': 'reach'}
        # if need to fully include the 1st n.n., then 1 supercells are needed
        n = supercell_count(**options)
        self.assertEqual(n, 0)
        # for 2nd n.n., 1 supercells are also needed
        n = supercell_count(**(options | {'nshell': 2}))
        self.assertEqual(n, 1)
        # for 3rd n.n., 1 supercells are also needed
        n = supercell_count(**(options | {'nshell': 3}))
        self.assertEqual(n, 1)
        # for 4th n.n., 1 supercells are also needed
        n = supercell_count(**(options | {'nshell': 4}))
        self.assertEqual(n, 1)
        # for 5th n.n., 1 supercells are also needed
        n = supercell_count(**(options | {'nshell': 5}))
        self.assertEqual(n, 2)
    
    def test_supercell_count_full(self):
        # test with one conventional cell of sc
        options = {'abc': [4, 4, 4], 'nshell': 1, 'atoms': [[0, 0, 0]], 'mode': 'full'}
        # for 1st n.n., 2 layers are needed: CN = 6
        n = supercell_count(**options)
        self.assertEqual(n, 2)
        # for 2nd n.n., 2 layers are needed: CN = 12
        n = supercell_count(**(options | {'nshell': 2}))
        self.assertEqual(n, 2)
        # for 3rd n.n., 2 layers are needed: CN = 8
        n = supercell_count(**(options | {'nshell': 3}))
        self.assertEqual(n, 2)
        # for 4th n.n., 4 layers are needed: CN = 6
        n = supercell_count(**(options | {'nshell': 4}))
        self.assertEqual(n, 4)
        # for 5th n.n., 4 layers are needed: CN = 24
        n = supercell_count(**(options | {'nshell': 5}))
        self.assertEqual(n, 4)
        
        # test with one conventional cell of bcc, because there are more than one
        # atoms in cell, even number of layers may enter
        options = {'abc': [2.86304]*3, 'nshell': 1, 'atoms': [[0, 0, 0], [0.5, 0.5, 0.5]], 
                   'mode': 'full'}
        # for 1st n.n., 1 layers are needed: CN = 8
        n = supercell_count(**options)
        self.assertEqual(n, 1)
        # for 2nd n.n., 2 layers are needed: CN = 6
        n = supercell_count(**(options | {'nshell': 2}))
        self.assertEqual(n, 2)
        # for 3rd n.n., 2 layers are needed: CN = 12
        n = supercell_count(**(options | {'nshell': 3}))
        self.assertEqual(n, 2)
        # for 4th n.n., 3 layers are needed: CN = 24
        n = supercell_count(**(options | {'nshell': 4}))
        self.assertEqual(n, 3)
        # for 5th n.n., 3 layers are needed: CN = 8
        n = supercell_count(**(options | {'nshell': 5}))
        self.assertEqual(n, 3)
    
    def test_pbcdist(self):
        cell = np.diag([1]*3)
        atoms = [[0, 0, 0], [0.3, 0, 0], [0.7, 0, 0]]
        dist = pbcdist(cell, this=[0, 0, 0], other=atoms)
        self.assertTrue(np.allclose(dist, [0, 0.3, 0.3])) # 0 -> 0.7 fold to 0.3
    
if __name__ == '__main__':
    
    test = init()
    unittest.main(exit=test)
    