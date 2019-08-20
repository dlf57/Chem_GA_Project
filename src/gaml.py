'''
ML code for GA
'''
from sklearn.linear_model import BayesianRidge
from modAL.models import ActiveLearner
import oddt
from oddt.fingerprints import ECFP
import pybel
import pickle
import pandas as pd
import numpy as np


def representation(rep_str, polymer, poly_size, smiles_list):
    '''
    Calls desired machine learning representation function

    Parameters
    ----------
    rep_str: str
        which machine learning representation
    polymer: list (specific format)
        [(#,#,#,#), A, B]
    poly_size: int
        number of monomers per polymer
    smiles_list: list
        list of all possible monomer SMILES

    Returns
    -------
    rep: array
        representation for that molecule
    '''
    accepted_reps = ['ECFP', 'MONOSEQ']
    if rep_str == 'ECFP':
        rep = ecfp(polymer, poly_size, smiles_list)
    elif rep_str == 'MONOSEQ':
        rep = monoseq(polymer, poly_size)
    else:
        accept_reps = str(accepted_reps).strip('[]')
        raise NotImplementedError(
            'Representation \'{}\' is unsupported. Accepted representations are {} .'.format(rep_str, accept_reps))

    return rep


def ecfp(polymer, poly_size, smiles_list, depth=2, size=4096):
    '''
    Create ECFP representation

    Parameters
    ----------
    polymer: list (specific format)
        [(#,#,#,#), A, B]
    poly_size: int
        number of monomers per polymer
    smiles_list: list
        list of all possible monomer SMILES
    depth: int
        depth of the fingerprint (eg. 2 is ECFP4)
    size: int
        size of representation

    Returns
    -------
    rep: array
        ECFP representation for that molecule
    '''
    # make polymer into SMILES string
    poly_smiles = make_polymer_str(polymer, smiles_list, poly_size)

    mol = oddt.toolkit.readstring("smi", poly_smiles)
    fp = ECFP(mol, depth=depth, size=size, sparse=False)
    fpl = list(fp)

    rep = []
    for i in range(len(fpl)):
        rep.append(fpl[i])

    return np.asarray(rep, dtype=np.int)


def monoseq(polymer, poly_size):
    '''
    Creates a representation based on the monomers and the sequence
    .. eg. [127, 983, 1, 0, 1, 0, 0, 0]

    Parameters
    ----------
    polymer: list (specific format)
        [(#,#,#,#), A, B]
    poly_size: int
        number of monomers per polymer

    Returns
    -------
    rep: array
        MONOSEQ representation for that molecule
    '''
    m1 = polymer[1]
    m2 = polymer[2]
    seq = list(polymer[0] * ((poly_size // 4) + 1))[:poly_size]
    rep = [m1, m2] + seq
    rep = np.asarray(rep, dtype=np.int)

    return rep


def load_regressor(stored_regr):
    '''
    Loads pickled regressor

    Paramaters
    ----------
    stored_regr: str
        path to stored regressor

    Returns
    -------
    loaded_regr: obj
        trained regressor
    '''
    with open(stored_regr, 'rb') as regrfile:
        loaded_regr = pickle.load(regrfile)

    return loaded_regr


def store_regressor(storage, regr):
    '''
    Loads pickled regressor

    Paramaters
    ----------
    storage: str
        path to store regressor
    regr: obj
        trained regressor
    '''
    with open(storage, 'wb') as regrfile:
        pickle.dump(regr, regrfile)


def reteach(X_update, y_update, regr):
    '''
    Updates the regressor with the new data

    Paramaters:
    ----------
    X_update: list
        representation list of new molecules
    y_update: list
        list of new molecules properties calculed w/ GFN2
    regr: obj
        machine learning regressor to be updated

    Returns: obj
        updated machine learning regressor
    '''
    reteach = pd.DataFrame({'X': X_update, 'y': y_update})
    # delete values w/ -10 as those are failed files
    reteach = reteach[reteach.y != -10]
    X_reteach = np.asarray(list(reteach['X']))
    y_reteach = np.asarray(list(reteach['y']))
    regr.teach(X_reteach, y_reteach)

    return regr
