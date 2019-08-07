'''
Notes:
-each polymer is defined in the format [(numerical sequence), monomer1 index, ... monomerX index]
-population is list of these polymers
'''

import os
import shutil
import subprocess
import string
import random
import math
import pybel
import numpy as np
from scipy import stats
from statistics import mean
from itertools import product
from copy import deepcopy
import shlex
import pandas as pd
#import forkmap

ob = pybel.ob

def make3D(mol):
    '''
    Makes the mol object from SMILES 3D

    Parameters
    ---------
    mol: object
        pybel molecule object
    '''
    # make mol object 3D and add hydrogens
    pybel._builder.Build(mol.OBMol)
    mol.addh()

    ff = pybel._forcefields["mmff94"]
    success = ff.Setup(mol.OBMol)
    if not success:
        ff = pybel._forcefields["uff"]
        success = ff.Setup(mol.OBMol)
        if not success:
            sys.exit("Cannot set up forcefield")

    ff.ConjugateGradients(100, 1.0e-3)
    ff.WeightedRotorSearch(100, 25)
    ff.ConjugateGradients(250, 1.0e-4)

    ff.GetCoordinates(mol.OBMol)
    
    
def find_sequences(num_mono_species):
    '''
    Finds all possible sequences

    Parameters
    ---------
    num_mono_species: int
        number of monomer species in each polymer (e.g. copolymer = 2)

    Returns
    -------
    numer_seqs: list
        all possible sequences as a numerical list
    '''
    # find length of each sequence
    seq_len = num_mono_species**2

    # find all possible sequences as numerical lists [start index 0]
    # (use cartesian product of each type of monomer over sequence length)
    numer_seqs = list(product(range(num_mono_species), repeat=seq_len))

    return numer_seqs

def find_poly_mw(population, poly_size, smiles_list):
    '''
    Calculates molecular weight of polymers

    Parameters
    ---------
    population: list
        list of polymers in population
    poly_size: int
        number of monomers per polymer
    smiles_list: list
        list of all possible monomer SMILES

    Returns
    -------
    poly_mw_list: list
        list of molecular weights of polymers in population
    '''
    poly_mw_list = []
    for polymer in population:
        # make polymer into SMILES string
        poly_smiles = make_polymer_str(polymer, smiles_list, poly_size)
        # make polymer string into pybel molecule object
        mol = pybel.readstring('smi', poly_smiles)

        # add mw of polymer to list
        poly_mw = mol.molwt
        poly_mw_list.append(poly_mw)

    return poly_mw_list

def make_file_name(polymer, poly_size):
    '''
    Makes file name for a given polymer

    Parameters
    ---------
    polymer: list (specific format)
        [(#,#,#,#), A, B]

    Returns
    -------
    file_name: str
        polymer file name (w/o extension) showing monomer indicies and full sequence
        e.g. 100_200_101010 for a certain hexamer
    '''

    # capture monomer indexes and numerical sequence as strings for file naming
    mono1 = str(polymer[1])
    mono2 = str(polymer[2])
    seq = polymer[0]

    # make string of actual length sequence (e.g. hexamer when poly_size = 6)
    triple_seq = seq + seq + seq
    true_length_seq = ''.join(str(triple_seq[i]) for i in range(poly_size))

    # make file name string
    file_name = '%s_%s_%s' % (mono1, mono2, true_length_seq)

    return file_name

def run_geo_opt(polymer, poly_size, smiles_list):
    '''
    Runs geometry optimization calculation on given polymer

    Parameters
    ---------
    polymer: list (specific format)
        [(#,#,#,#), A, B]
    poly_size: int
        number of monomers per polymer
    smiles_list: list
        list of all possible monomer SMILES

    '''
    # make file name string w/ convention monoIdx1_monoIdx2_fullNumerSequence
    file_name = make_file_name(polymer, poly_size)
    
    #if output file already exists, skip xTB
    exists = os.path.isfile('output/%s.out' % (file_name))
    if exists:
        print("output file existed")
        return
            
    # make polymer into SMILES string
    poly_smiles = make_polymer_str(polymer, smiles_list, poly_size)

    # make polymer string into pybel molecule object
    mol = pybel.readstring('smi', poly_smiles)
    make3D(mol)

    # write polymer .xyz file to containing folder
    mol.write('xyz', 'input/%s.xyz' % (file_name), overwrite=True)

    # run xTB geometry optimization
    xtb = subprocess.call('(/ihome/ghutchison/geoffh/xtb/xtb input/%s.xyz --opt >output/%s.out)' % (file_name, file_name), shell=True)

    save_opt_file = subprocess.call('(cp xtbopt.xyz opt/%s_opt.xyz)' % (file_name), shell=True)
    del_restart = subprocess.call('(rm -f *restart)', shell=True)

def find_elec_prop(population, poly_size, smiles_list):
    '''
    Calculates dipole moment and polarizability of each polymer in population
    TODO: add dipole tensor functionality
    TODO: update parser to catch failures/errors in output file

    Parameters
    ---------
    population: list
        list of polymers in population
    poly_size: int
        number of monomers per polymer
    smiles_list: list
        list of all possible monomer SMILES

    Returns
    -------
    elec_prop_lists: list
        nested list of [list of polymer dipole moments, list of polymer polarizabilities]
    '''

    poly_polar_list = []
    poly_dipole_list = []
    
    #run xTB geometry optimization
    #nproc = 8
    for polymer in population:
        #forkmap.map(run_geo_opt, polymer, poly_size, smiles_list, n=nproc)
        run_geo_opt(polymer, poly_size, smiles_list)

    # parse xTB output files
    for polymer in population:   
         # make file name string w/ convention monoIdx1_monoIdx2_fullNumerSequence
        file_name = make_file_name(polymer, poly_size)
        
        # check for xTB failures
        if 'FAILED!' in open('output/%s.out' % (file_name)).read():
            # move output file to 'failed' directory
            move_fail_file = subprocess.call('(mv output/%s.out failed/%s.out)' % (file_name, file_name), shell=True)

            # note failure by filling property lists with dummy values
            poly_polar_list.append(-10)
            poly_dipole_list.append(-10)

        # if xTB successful, parse output file for static polarizability and dipole moment
        else:
            read_output = open('output/%s.out' % (file_name), 'r')
            for line in read_output:
                # create list of tokens in line
                tokens = line.split()

                if line.startswith(" Mol. α(0)"):
                    temp_polar = float(tokens[4])
                    poly_polar_list.append(temp_polar)
                elif line.startswith("   full:"):
                    # dipole tensor - STILL A LIST OF STRINGS (not floats)
                    # TODO: add tensor functionality later
                    dipole_line = tokens
                    temp_dipole = float(tokens[4])
                    poly_dipole_list.append(temp_dipole)
                    # break inner for loop to avoid overwriting with other lines starting with "full"
                    break
            read_output.close()

    # make nested list of dipole moment and polarizability lists
    elec_prop_lists = [poly_dipole_list, poly_polar_list]

    return elec_prop_lists

def fitness_fn(opt_property, poly_property_list):
    '''
    Ranks polymers in population based on associated property values
    Rank 0 = best

    Parameters
    ---------
    opt_property: str
        property being optimized
    poly_prop_list: list
        list of property values corresponding to polymers in population

    Returns
    -------
    ranked_indicies: list
        list of ranked polymer indicies with first entry being best
    '''
    # rank polymers based on highest property value = best
    if opt_property == "mw" or "dip":
        # make list of indicies of polymers in population, sorted based on property values
        ranked_indicies = list(np.argsort(poly_property_list))
        # reverse list so highest property value = 0th
        ranked_indicies.reverse()
    else:
        print("Error: opt_property not recognized. trace:fitness_fn")

    return ranked_indicies

def parent_select(opt_property, population, poly_property_list):
    '''
    Finds top half of population

    Parameters
    ---------
    opt_property: str
        property being optimized
    population: list
        list of polymers in population
    poly_prop_list: list
        list of properties of the polymers

    Returns
    -------
    parent_list: list
        top half of population
    '''
    # find number of parents (half of population)
    parent_count = int(len(population) / 2)

    # make list of ranked polymer indicies
    fitness_list = fitness_fn(opt_property, poly_property_list)

    # make list of top polymers
    parent_list = []
    for x in range(parent_count):
        parent_list.append(population[fitness_list[x]])

    return parent_list

def mutate(polymer, sequence_list, smiles_list, mono_list):
    '''
    Creates point mutation in given polymer if polymer is selected for mutation

    Parameters
    ---------
    polymer: list (specific format)
        [(#,#,#,#), A, B]
    sequence_list: list
        list of sequences
    smiles_list: list
        list of monomer SMILES

    Returns
    -------
    polymer: list (specific format)
        [(#,#,#,#), A, B]
    '''

    # set mutation rate
    mut_rate = 0.4

    # determine whether to mutate based on mutation rate
    rand = random.randint(1,10)
    if rand <= (mut_rate*10):
        pass
    else:
        return polymer

    # choose point of mutation (sequence or specific monomer)
    point = random.randint(0, len(polymer) - 1)
    # replace sequence
    if point == 0:
        polymer[point] = sequence_list[random.randint(
            0, len(sequence_list) - 1)]
    # or replace specific monomer
    else:
        new_mono = random.randint(0, len(smiles_list) - 1)
        #new_mono = weighted_random_pick(mono_list)
        polymer[point] = new_mono
        # increase frequency count for monomer in mono_list
        mono_list[new_mono][1] += 1

    return polymer

def weighted_random_pick(mono_list):
    '''
    Selects random monomer based on weighted (frequency) system

    Parameters
    ---------
    mono_list: list (specific format)
        [[monomer index, frequency]]

    Returns
    -------
    mono_idx: int
        index of selected randomly chosen monomer
    '''

    ordered_mono_list = deepcopy(mono_list)

    # sort based on frequencies, put in ASCENDING order (LOWEST first)
    ordered_mono_list = sorted(ordered_mono_list, key = lambda monomer: monomer[1])

    # calculate sum of all weights [frequencies]
    sum = 0
    for idx in ordered_mono_list:
        freq = idx[1]
        sum += freq

    # generate random number from 0 to sum, lower endpoint included
    rand_num = random.randint(0, sum-1)

    # loop over list of sorted weights, subtract weights from random number until random number is less than next weight
    for idx in ordered_mono_list:
        mono_idx = idx[0]
        freq = idx[1]
        if rand_num < freq:
            return mono_idx
        else:
            rand_num -= freq

def crossover_mutate(parent_list, pop_size, poly_size, num_mono_species, sequence_list, smiles_list, mono_list):
    '''
    Performs crossover and mutation functions on given population
    TODO: fix possible duplication problem after mutation

    Parameters
    ---------
    parent_list: list
        list of parent polymers
    pop_size: int
        number of polymers in each generation
    num_mono_species: int
        number of monomer species in each polymer (e.g. copolymer = 2)
    sequence_list: list
        list of sequences
    smiles_list: list
        list of all possible monomer SMILES

    Returns
    -------
    new_pop: list
        population list after crossover and mutation
    '''

    # initialize new population with parents
    new_pop = deepcopy(parent_list)
    new_pop_str = []
    for parent in new_pop:
        parent_str = make_polymer_str(parent, smiles_list, poly_size)
        new_pop_str.append(parent_str)

    # loop until enough children have been added to reach population size
    while len(new_pop) < pop_size:

        # randomly select two parents (as indexes from parent list) to cross
        parent_a = random.randint(0, len(parent_list) - 1)
        parent_b = random.randint(0, len(parent_list) - 1)

        # ensure parents are unique indiviudals
        if len(parent_list) > 1:
            while parent_b == parent_a:
                parent_b = random.randint(0, len(parent_list) - 1)

        # determine number of monomers taken from parent A
        num_mono_a = random.randint(1, num_mono_species)

        # randomly determine which parent's sequence will be used
        par_seq = random.randint(0, 1)

        # create hybrid child
        temp_child = []

        # give child appropriate parent's sequence
        if par_seq == 0:
            temp_child.append(parent_list[parent_a][0])
        else:
            temp_child.append(parent_list[parent_b][0])

        # give child first half monomers from A, second half from B
        for monomer in range(1, num_mono_a + 1):
            temp_child.append(parent_list[parent_a][monomer])
        if num_mono_a < num_mono_species:
            for monomer in range(num_mono_a + 1, num_mono_species + 1):
                temp_child.append(parent_list[parent_b][monomer])

        # give child opportunity for mutation
        temp_child = mutate(temp_child, sequence_list, smiles_list, mono_list)

        temp_child_str = make_polymer_str(temp_child, smiles_list, poly_size)

        #try to avoid duplicates in population, but prevent infinite loop if unique individual not found after so many attempts
        # TODO: fix possible duplication problem after mutation
        if temp_child_str in new_pop_str:
            pass
        else:
            new_pop.append(temp_child)
            new_pop_str.append(temp_child_str)

    return new_pop

def sort_mono_indicies_list(mono_list):
    '''
    Makes list of all monomer indicies ordered by frequency (highest first)

    Parameters
    ---------
    mono_list: list (specific format)
        [[monomer index, frequency]]

    Returns
    -------
    mono_indicies: list
        list of monomer indicies ordered by frequency (highest first)
    '''

    ordered_mono_list = deepcopy(mono_list)

    # sort based on frequencies, put in descending order (highest first)
    ordered_mono_list = sorted(ordered_mono_list, key = lambda monomer: monomer[1], reverse = True)

    # make list of monomer indicies only (in order of descending frequency)
    mono_indicies = []
    for x in range(len(ordered_mono_list)):
        mono_indicies.append(ordered_mono_list[x][0])

    return mono_indicies

def make_polymer_str(polymer, smiles_list, poly_size):
    '''
    Constructs polymer string from monomers, adds standard end groups [amino and nitro]

    Parameters
    ---------
    polymer: list (specific format)
        [(#,#,#,#), A, B]
    smiles_list: list
        list of all possible monomer SMILES
    poly_size: int
        number of monomers per polymer

    Returns
    -------
    poly_string: str
        polymer SMILES string
    '''
    poly_string = ''

    # add amino end group
    poly_string = poly_string + "N"

    # cycle over monomer sequence until total number of monomers in polymer is reached
    cycle = 0
    for i in range(poly_size):
        seq_cycle_location = i - cycle * (len(polymer[0]))
        seq_monomer_index = polymer[0][seq_cycle_location]
        if seq_cycle_location == (len(polymer[0]) - 1):
            cycle += 1

        # find monomer index from given sequence location in polymer
        monomer_index = polymer[seq_monomer_index + 1]
        poly_string = poly_string + smiles_list[monomer_index]

    #add nitro end group
    poly_string = poly_string + "N(=O)=O"

    return poly_string


def main():
    # number of polymers in population
    pop_size = 32
    # number of monomers per polymer
    poly_size = 6
    # number of species of monomers in each polymer
    num_mono_species = 2
    # number of all possible monomers in imported monomer list
    mono_list_size = 1234

    # property of interest (options: molecular weight 'mw', dipole moment 'dip')
    opt_property = "pol"

    # Read in monomers from input file
    # read_file = open('../input_files/1235MonomerList.txt', 'r')
    read_file = open('/ihome/ghutchison/dch45/Chem_GA_Project/input_files/1235MonomerList.txt', 'r')

    # create list of monomer SMILES strings
    # assumes input file has one monomer per line
    smiles_list = []
    for line in read_file:
        # grab first token from each line
        temp = line.split()[0]
        smiles_list.append(temp)
    read_file.close()

    #create monomer frequency list [(mono index 1, frequency), (mono index 2, frequency),...]
    mono_list = []
    for x in range(len(smiles_list)):
        mono_list.append([x, 0])

    # create all possible numerical sequences for given number of monomer types
    sequence_list = find_sequences(num_mono_species)

    # create inital population as list of polymers
    population = []
    population_str = []
    counter = 0
    while counter < pop_size:
        # for polymer in range(pop_size):
        temp_poly = []

        # select sequence type for polymerf
        poly_seq = sequence_list[random.randint(0, len(sequence_list) - 1)]
        temp_poly.append(poly_seq)

        # select monomer types for polymer
        for num in range(num_mono_species):
            # randomly select a monomer index
            poly_monomer = random.randint(0, len(smiles_list) - 1)
            temp_poly.append(poly_monomer)
            # increase frequency count for monomer in mono_list
            mono_list[poly_monomer][1] += 1

        # make SMILES string of polymer
        temp_poly_str = make_polymer_str(temp_poly, smiles_list, poly_size)

        # add polymer to population
        # check for duplication - use str for comparison to avoid homopolymer, etc. type duplicates
        if temp_poly_str in population_str:
            pass
        else:
            population.append(temp_poly)
            population_str.append(temp_poly_str)
            counter += 1


    # find initial population properties
    if opt_property == 'mw':
        # calculate MW for each monomer
        mw_list = []
        for monomer in smiles_list:
            temp_mono = pybel.readstring("smi", monomer).molwt
            mw_list.append(temp_mono)
        # Calculate polymer molecular weights
        poly_property_list = find_poly_mw(population, poly_size, smiles_list)
    elif opt_property == 'dip':
        #initialize list of polarizabilities
        polar_list = []
        # calculate electronic properties for each polymer
        elec_prop_list = find_elec_prop(population, poly_size, smiles_list)
        poly_property_list = elec_prop_list[0]
        polar_list = elec_prop_list[1]
    elif opt_property == 'pol':
        #initialize list of dipole moments
        dip_list = []
        # calculate electronic properties for each polymer
        elec_prop_list = find_elec_prop(population, poly_size, smiles_list)
        poly_property_list = elec_prop_list[1]
        dip_list = elec_prop_list[0]
        
    else:
        print("Error: opt_property not recognized. trace:main:initial pop properties")



    # set initial values for min, max, and avg polymer weights
    min_test = min(poly_property_list)
    max_test = max(poly_property_list)
    avg_test = mean(poly_property_list)

    if opt_property == 'dip':
        compound = make_file_name(population[poly_property_list.index(max_test)], poly_size)
        polar_val = polar_list[poly_property_list.index(max_test)]
        
    if opt_property == 'pol':
        compound = make_file_name(population[poly_property_list.index(max_test)], poly_size)
        dip_val = dip_list[poly_property_list.index(max_test)]

    # create new output files
    analysis_file = open('gens_analysis.txt', 'w+')
    population_file = open('gens_population.txt', 'w+')
    values_file = open('gens_values.txt', 'w+')
    if opt_property == 'dip':
        dip_polar_file = open('gens_dip_polar.txt', 'w+')
    if opt_property == 'pol':
        polar_dip_file = open('gens_polar_dip.txt', 'w+')
    spear_file = open('gens_spear.txt', 'w+')

    # write files headers
    analysis_file.write('min, max, avg, spearman, \n')
    population_file.write('polymer populations \n')
    values_file.write('%s values \n' % (opt_property))
    if opt_property == 'dip':
        dip_polar_file.write('compound, gen, dipole, polar \n')
    if opt_property == 'pol':
        polar_dip_file.write('compound, gen, polar, dip \n')
    spear_file.write('gen, spear_05, spear_10, spear_15 \n')

    #capture initial population data
    analysis_file.write('%f, %f, %f, n/a, \n' % (min_test, max_test, avg_test))
    if opt_property == 'dip':
        dip_polar_file.write('%s, %d, %f, %f, \n' % (compound, 1, max_test, polar_val))
    if opt_property == 'pol':
        polar_dip_file.write('%s, %d, %f, %f, \n' % (compound, 1, max_test, dip_val))
    spear_file.write('1, n/a, n/a, n/a, \n')

    # write polymer population to file
    for polymer in population:
        poly_name = make_file_name(polymer, poly_size)
        population_file.write('%s, ' % (poly_name))
    population_file.write('\n')

    for value in poly_property_list:
        values_file.write('%f, ' % (value))
    values_file.write('\n')
    
    # close all output files
    analysis_file.close()
    population_file.close()
    values_file.close()
    if opt_property == 'dip':
        dip_polar_file.close()
    if opt_property == 'pol':
        polar_dip_file.close()
    spear_file.close()
    
    # make backup copies of output files
    shutil.copy('gens_analysis.txt', 'gens_analysis_copy.txt')
    shutil.copy('gens_population.txt', 'gens_population_copy.txt')
    shutil.copy('gens_values.txt', 'gens_values_copy.txt')
    if opt_property == 'dip':
        shutil.copy('gens_dip_polar.txt', 'gens_dip_polar_copy.txt')
    if opt_property == 'pol':
        shutil.copy('gens_polar_dip.txt', 'gens_polar_dip_copy.txt')
    shutil.copy('gens_spear.txt', 'gens_spear_copy.txt')

    # Loop

    # check for convergence among top 30% (or top 8, whichever is larger) candidates between 5 generations
    perc = 0.1
    n = int(mono_list_size*perc)
    n_05 = int(mono_list_size*.05)
    n_10 = int(mono_list_size*.10)
    n_15 = int(mono_list_size*.15)


    # initialize generation counter
    gen_counter = 1

    # initialize convergence counter
    spear_counter = 0
    prop_value_counter = 0

    #while spear_counter < 10 or prop_value_counter < 10:
    for x in range(100):
        # open output files
        analysis_file = open('gens_analysis.txt', 'a+')
        population_file = open('gens_population.txt', 'a+')
        values_file = open('gens_values.txt', 'a+')
        if opt_property == 'dip':
            dip_polar_file = open('gens_dip_polar.txt', 'a+')
        if opt_property == 'pol':
            polar_dip_file = open('gens_polar_dip.txt', 'a+')
        spear_file = open('gens_spear.txt', 'a+')
        
        
        gen_counter += 1
        
        max_init = max(poly_property_list)

        # create sorted monomer list with most freq first
        gen1 = sort_mono_indicies_list(mono_list)

        # Selection - select heaviest (best) 50% of polymers as parents
        population = parent_select(opt_property, population, poly_property_list)

        # Crossover & Mutation - create children to repopulate bottom 50% of polymers in population
        population = crossover_mutate(population, pop_size, poly_size, num_mono_species, sequence_list, smiles_list, mono_list)

        # calculate desired polymer property
        if opt_property == "mw":
            poly_property_list = find_poly_mw(population, poly_size, smiles_list)
        elif opt_property == "dip":
            elec_prop_list = find_elec_prop(population, poly_size, smiles_list)
            poly_property_list = elec_prop_list[0]
            polar_list = elec_prop_list[1]
        elif opt_property == 'pol':
            elec_prop_list = find_elec_prop(population, poly_size, smiles_list)
            poly_property_list = elec_prop_list[1]
            dip_list = elec_prop_list[0]
        else:
            print("Error: opt_property not recognized. trace:main:loop pop properties")

        # record representative generation properties
        min_test = min(poly_property_list)
        max_test = max(poly_property_list)
        avg_test = mean(poly_property_list)

        if opt_property == 'dip':
            compound = make_file_name(population[poly_property_list.index(max_test)], poly_size)
            polar_val = polar_list[poly_property_list.index(max_test)]
            
        if opt_property == 'pol':
            compound = make_file_name(population[poly_property_list.index(max_test)], poly_size)
            dip_val = dip_list[poly_property_list.index(max_test)]

        # create sorted monomer list with most freq first
        gen2 = sort_mono_indicies_list(mono_list)

        # calculate Spearman correlation coefficient for begin and end sorted monomer lists
        spear = stats.spearmanr(gen1[:n], gen2[:n])[0]

        spear_05= stats.spearmanr(gen1[:n_05], gen2[:n_05])[0]
        spear_10 = stats.spearmanr(gen1[:n_10], gen2[:n_10])[0]
        spear_15 = stats.spearmanr(gen1[:n_15], gen2[:n_15])[0]

        # capture monomer indexes and numerical sequence as strings for population file
        analysis_file.write('%f, %f, %f, %f, \n' % (min_test, max_test, avg_test, spear))
        if opt_property == 'dip':
            dip_polar_file.write('%s, %d, %f, %f, \n' % (compound, gen_counter, max_test, polar_val))
        if opt_property == 'pol':
            polar_dip_file.write('%s, %d, %f, %f, \n' % (compound, gen_counter, max_test, dip_val))
        spear_file.write('%d, %f, %f, %f, \n' % (gen_counter, spear_05, spear_10, spear_15))


        # write polymer population to file
        for polymer in population:
            poly_name = make_file_name(polymer, poly_size)
            population_file.write('%s, ' % (poly_name))
        population_file.write('\n')

        for value in poly_property_list:
            values_file.write('%f, ' % (value))
        values_file.write('\n')

        # keep track of number of successive generations meeting Spearman criterion
        if spear > 0.92:
            spear_counter += 1
        else:
            spear_counter = 0
            
        # keep track of number of successive generations meeting property value convergence criterion
        if max_test >= (max_init - max_init*0.05) and max_test <= (max_init + max_init*0.05):
            prop_value_counter += 1
        else:
            prop_value_counter = 0
            
        # close all output files
        analysis_file.close()
        population_file.close()
        values_file.close()
        if opt_property == 'dip':
            dip_polar_file.close()
        if opt_property == 'pol':
            polar_dip_file.close()
        spear_file.close()
        
        # make backup copies of output files
        shutil.copy('gens_analysis.txt', 'gens_analysis_copy.txt')
        shutil.copy('gens_population.txt', 'gens_population_copy.txt')
        shutil.copy('gens_values.txt', 'gens_values_copy.txt')
        if opt_property == 'dip':
            shutil.copy('gens_dip_polar.txt', 'gens_dip_polar_copy.txt')
        if opt_property == 'pol':
            shutil.copy('gens_polar_dip.txt', 'gens_polar_dip_copy.txt')
        shutil.copy('gens_spear.txt', 'gens_spear_copy.txt')

    #remove unnecessary copies
    del_copies = subprocess.call('(rm -f *_copy.txt)', shell=True)



if __name__ == '__main__':
    main()
