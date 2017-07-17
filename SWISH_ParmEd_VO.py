#!/home/vladas/software/anaconda/bin/python

d = """
===============================================================================
Modification of Amber system topology to set up simulations for
SWISH (Sampling Water Interfaces through Scaled Hamiltonians) method.

Modifies LJ pair potentials between OW (water) and apolar (C, S) atomtypes by
linear scaling in replicas, optionally, only for preferred residues.

In addition, now allows switching modifying inter ligand interactions to prevent
aggregation (tuned repulsive LJ between ligand heavy atoms).

Developed for Amber type of protein force-fields. Assumes ligand atomtypes
are in lower cases (GAFF format).
------------------------
Example:
./SWISH_ParmEd.py -f <prmtop_file> -nreps 6 -smax 1.5 -hmr -v
takes as input amber topology file <prmtop_file> to produce 6 replicas with
SWISH scaling factor up to 1.5; and default HMassRepartition by parmed.tools.

The script relies on ParmEd and Numpy tools.

Vladas Oleinikovas
uccavol@ucl.ac.uk

version beta.4                                                    24/10/2016
================================================================================
"""



import os
import copy
import sys
import argparse
import numpy as np
import parmed
from parmed.constants import NTYPES
import time

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=d, epilog=" ")

parser.add_argument("-f", type=str, default='system.wat.leap.prmtop', help='input AMBER topology file name (default: %(default)s)')
parser.add_argument("-o", type=str, default=None, help='output AMBER topology file name (default: system[HMR]_SWISH.XXX.prmtop)')
parser.add_argument("-nreps", type=int, default=6, help='number of replicas (default: %(default)s)')
parser.add_argument("-smax", type=float, default=1.5, help="maximum scaling factor (default: %(default)s)")
parser.add_argument("-smin", type=float, default=1.0, help="minimum scaling factor (default: %(default)s)")
parser.add_argument("-scale", type=float, nargs='+', default=[], help="list of scaling factor(s); overrides -nreps, -smax, -smin (default: %(default)s)")
parser.add_argument("-ignore", default=False, action='store_true', help="ignore warnings (default: %(default)s)")
parser.add_argument("-frag_repel", default=False, action='store_true', help="use inter ligand repulsion; (default: %(default)s)")
parser.add_argument("-hmr", default=False, action='store_true', help="HMassRepartition (default: %(default)s)")
parser.add_argument("-hm_Da", type=float, default=None, help="H mass in Daltons used in HMR (default: use parmed default, 3.024)")
parser.add_argument("-v", "--verbose", action='store_true', help="be verbose")
parser.add_argument("-xyz", type=str, default='system.wat.leap.rst7', help='input AMBER coordinate file name (default: %(default)s)')
parser.add_argument("-gmx", default=False, action='store_true', help="save GROMACS topology, too (default: %(default)s)")
parser.add_argument("-bind_pref", default=False, action='store_true', help="SWISH scaling to preferred binding residues only (default: %(default)s)")


########################################################
# PARSING INPUTS
########################################################

args = parser.parse_args()

inputfile = args.f
outputname = args.o
nreps = args.nreps
smax = args.smax
smin = args.smin
scale = [ sf for sf in args.scale ]
HMR = args.hmr
HM_Da = args.hm_Da
frag_repel = args.frag_repel
verbose = args.verbose
ignore = args.ignore
inputcoords = args.xyz
gmx = args.gmx
bind_pref = args.bind_pref

#----------------------------------------------
def get_ligand_atomtypes(input_parm, hydrogens=False):
    """
    Read all atoms and returns a dictionary of ligand atoms.
    """
    # select ligand atoms that will want to repel
    lig_atomtypes = []

    for atom in input_parm.atoms:
        if atom.type not in lig_atomtypes:
            # non-protein atoms; ligands are lower-cases in GAFF
            if atom.type.islower():
                if hydrogens: # add without checking
                    lig_atomtypes.append(atom.type)
                elif atom.atomic_number > 1: # exclude hydrogens and dummies
                    lig_atomtypes.append(atom.type)
    return lig_atomtypes

#----------------------------------------------
def get_apolar_atomtypes(input_parm):
    """
    Read all residues and atoms and return a list of sidechain (+ C_alpha)
    C and S atomtypes to be used in SWISH scaling.
    """
    # apolar atomtypes
    apolar_atomtypes = []
    for resid in input_parm.residues:
        for atom in resid.atoms:
            if atom.atomic_number in [6, 16]: # C and S
                # exclude backbone carbonyl carbon - named 'C'!
                # avoid non-protein atoms; ligands are lower-cases in GAFF
                if atom.name != 'C' and atom.type.isupper() and atom.type not in apolar_atomtypes:
                    apolar_atomtypes.append(atom.type)
    return apolar_atomtypes

#----------------------------------------------
def do_HMR(input_parm, HM_Da):
    """
    Perform H-Mass Repartition.
    HM_Da - hydrogen mass in Daltons (float).
    """
    act_HMR = parmed.tools.actions.HMassRepartition(input_parm, arg_list=HM_Da)
    log.write(str(act_HMR)+"\n")
    if verbose: print(str(act_HMR))

    act_HMR.execute()
    
    input_parm.parm_comments['USER_COMMENTS'] += [str(act_HMR)]
    
    return input_parm

#---------------------------------------------
def add_interligand_repulsion(input_parm, lig_atomtypes, verbose=False):
    """
    Set interligang LJPair into an effectively repulsive potential
    excluding them from being within the same solvation shell
    (default setting: soft wall ~5A, decaying to 0 at sigma).
    Returns an updated parmed parameter object.
    """

    # GROMACS
    sigma = 0.8	#1.0 # nm
    epsilon = 1.5e-3	#1e-4 # kJ/mol
    gromacs_nonbond = "; interligand repulsion\n"

    # AMBER
    Rmin_ij = sigma *10 * 2**(-1.0/6) # convert to Rmin in A
    eps_ij = epsilon / 4.184 #convert to epsilon in kcal/mol

    # select atomtnames for scaling atomtypes
    for i in np.arange(len(lig_atomtypes)):
        for j in np.arange(i, len(lig_atomtypes)):
            # use atype.replace("*", "\*") in selection to avoid wild-card results!
            # strangely "\*" works instead of "'*'" here, unlike in the ParmEd manual
            # on selection masks - http://parmed.github.io/ParmEd/html/parmed.html#atom-selection-masks
            type_i = lig_atomtypes[i].replace("*", "\\\*")
            type_j = lig_atomtypes[j].replace("*", "\\\*")
            # change interligand attraction into soft repulsion!
            act_changeLJPair = parmed.tools.actions.changeLJPair(input_parm, "@%" + type_i, "@%" + type_j, Rmin_ij, eps_ij)
            print(str(act_changeLJPair)+"\n")
            # execute
            act_changeLJPair.execute()

            # add equivalent line in GROMACS topology
            gromacs_nonbond += "%-10s   %-10s  1   %8.6e   %8.6e\n" % (type_i, type_j, sigma, epsilon)

    # add COMMENTS
    input_parm.parm_comments['USER_COMMENTS'] += ['Interligand repulsion set to (Rmin, eps) = (%.3e, %.3e)' % (Rmin_ij, eps_ij)]


    return input_parm, gromacs_nonbond


#----------------------------------------------
def get_residue_apolar_atom_dictionary(input_parm):
    """
    Read all residues and atoms and return a residue dictionary of
    sidechain (+ C_alpha) C and S atomnames with associated atomtypes.
    """
    # select atomtnames for scaling atomtypes
    residues_apolar = {}
    for resid in input_parm.residues:
        if resid.name not in residues_apolar.keys(): 
            atoms_apolar = {} # select atoms that will want to scale!
            for atom in resid.atoms:
                if atom.atomic_number in [6, 16]: # C and S
                    # exclude backbone carbonyl carbon - named 'C'!
                    # avoid non-protein atoms; ligands are lower-cases in GAFF
                    if atom.name != 'C' and not atom.type.islower():
                        atoms_apolar[atom.name] = atom.type
            residues_apolar[resid.name] = atoms_apolar
    return residues_apolar                          


#----------------------------------------------
def create_LJtypes(input_parm):
    """
    Rename selected C/S atomtypes from preferred residues by changing
    the AMBER_ATOM_TYPE to x+str(old_atom_name).
    New atomtype is then added to the LJ type lists (depth and radius
    values are inherited) and new nb_idx is assigned.
    Returns an updated parmed parameter object.
    """
    
    # get sidechain C/S atom_names and atom_types for each residue (includes C_alpha)
    residues_apolar = get_residue_apolar_atom_dictionary(input_parm)
    
    # preferred binding site residues!
    preferred = ['TYR', 'PHE', 'TRP', 'MET', 'HIS', 'ILE', 'LEU', 'GLY', 'CYS', 'VAL']
    
    if verbose:
        print("Preferred binding site residues:\n%s.\n" % ', '.join(preferred) + \
              "We will scale water interactions to their sidechain C/S atoms (+ C_alpha)!\n" + \
              "(to change, modify the definition in create_LJtypes function)\n")
    
    # include alternative protonation states of HIS!
    preferred += ['HIE', 'HID', 'HIP']
          
    # RENAME ATOMTYPES
    log.write("WARNING: Renaming the selected residue atom atomtypes!\n")
    atomtypes_changed = []
    for resname in residues_apolar.keys():
        if resname in preferred:
            for (aname, atype) in residues_apolar[resname].iteritems():
                # change selected atomtype - rename atomtypes!
                new_atype = "x%s" % (atype)
                # on selection masks - http://parmed.github.io/ParmEd/html/parmed.html#atom-selection-masks
                act_changetype = parmed.tools.change(input_parm, ":%s@%s" % (resname, aname), "AMBER_ATOM_TYPE %s" % new_atype)
                # write action to log
                log.write(str(act_changetype)+"\n")
                if list(act_changetype.mask.Selected()) == []:
                    print("WARNING: empty selection %s !\n" % act_changetype.mask.mask)
                    exit(3)
                
                act_changetype.execute()

                # keep track of renamed atomtypes
                if new_atype not in atomtypes_changed:
                    atomtypes_changed.append(new_atype)

    if verbose: print("Atomtypes renamed:\n%s\n" % ', '.join(atomtypes_changed))

    # ADD NEW ATOMTYPES
    log.write("WARNING: Adding the changed atomtypes (by name)!\n")
    for new_atype in atomtypes_changed:
        # add LJtype, based on updated atomtype name;
        # here is some overlap between atomtypes in different residues
        if new_atype not in input_parm.LJ_types.keys():
            # add LJtype - without specification - uses epsilon and Rmin/2 of previous atomtype (default)
            # updates atom.nb_idx and adds new entry to LJ_depth (espilon), LJ_radius (Rmin/2)
            old_nb_idx = input_parm.LJ_types[new_atype[1:]]
            old_rh, old_eps = input_parm.LJ_radius[old_nb_idx-1], input_parm.LJ_depth[old_nb_idx-1]

            # use atype.replace("*", "\*") in selection to avoid wild-card results!
            # strangely "\*" works instead of "'*'" here, unlike in the ParmEd manual
            # on selection masks - http://parmed.github.io/ParmEd/html/parmed.html#atom-selection-masks
            act_addLJtype = parmed.tools.addLJType(input_parm, "@%" + new_atype.replace("*", "\\\*") +\
                            " radius %s epsilon %s" % (old_rh, old_eps))
            # check due to past inconsistencies!
            if list(act_addLJtype.mask.Selected()) == []:
                print("WARNING: empty selection %s !\n" % act_addLJtype.mask.mask)
                exit(3)

            act_addLJtype.execute()

            # write action to log
            log.write(str(act_addLJtype)+"\n")

            # adds nb_idx - last index of LJ_depth / LJ_radius to LJ_types dictionary!
            new_nb_idx = len(input_parm.LJ_depth)
            input_parm.LJ_types[new_atype] = new_nb_idx

        else:
            print("WARNING: atomtype %s already exists!\n" % (new_atype))
            print input_parm.LJ_types.keys()
            exit(4)

    input_parm.parm_comments['USER_COMMENTS'] += ['New atomtypes created for preferred binding site residues:',\
						  ', '.join(preferred)]
                    
    return input_parm, atomtypes_changed

#----------------------------------------------
def combineLJPair(rh_i, eps_i, rh_j, eps_j):
    """
    LJ pair interaction Lorentz/Berthelot mixing rules for sigma and epsilon
    ----------
    r_h_i, r_h_j: 'float'
        half radii of i, j LJ Rmin
    eps_sr_i, eps_sr_j: 'float'
        square root of depth of i, j LJ potential well

    """
    Rmin_ij = (rh_i + rh_j)
    eps_ij = (eps_i * eps_j)**(0.5)
    return Rmin_ij, eps_ij


#----------------------------------------------
def scale_LJPairs(input_parm, atomtypes_changed, sfactor, verbose=False):
    """
    Multiplies water (atomtype = "OW") LJPair depth (epsilon) with
    selected atomtypes by sfactor according to SWISH scheme.
    Returns an updated parmed parameter object.
    """
    rh_OW = input_parm.LJ_radius[input_parm.LJ_types["OW"]-1] # Rmin/2
    eps_OW = input_parm.LJ_depth[input_parm.LJ_types["OW"]-1]

    # add COMMENTS
    input_parm.parm_comments['USER_COMMENTS'] += ['SWISH METHOD (scaling factor = %.3f)' % sfactor,\
                                                  'scaled OW LJ pair interactions with atomtypes',\
						  ', '.join(atomtypes_changed)]
    # FOR GROMACS
    gromacs_nonbond = "; SWISH scaling - %.2f\n" % sfactor

    # select atomtnames for scaling atomtypes
    for atype in atomtypes_changed:
        rh_i = input_parm.LJ_radius[input_parm.LJ_types[atype]-1] # Rmin/2
        eps_i = input_parm.LJ_depth[input_parm.LJ_types[atype]-1] # epsilon
        Rmin_iOW, eps_iOW = combineLJPair(rh_i, eps_i, rh_OW, eps_OW)
        eps_iOW *= sfactor # apply the scaling

        # use atype.replace("*", "\*") in selection to avoid wild-card results!
        # strangely "\*" works instead of "'*'" here, unlike in the ParmEd manual
        # on selection masks - http://parmed.github.io/ParmEd/html/parmed.html#atom-selection-masks
        act_changeLJPair = parmed.tools.actions.changeLJPair(input_parm, "@%" + atype.replace("*", "\\\*"), "@%OW", Rmin_iOW, eps_iOW)
        # write action to log
        if list(act_changeLJPair.mask1.Selected()) == []:
            print("WARNING: empty selection %s !\n" % act_changeLJPair.mask1.mask)
            exit(3)

        act_changeLJPair.execute()

        # add equivalent line in GROMACS topology
        sigma = Rmin_iOW * 2**(1.0/6) / 10 # convert to A
        epsilon = eps_iOW * 4.184     #convert to kJ/mol
        gromacs_nonbond += "%-10s   %-10s  1   %8.6e   %8.6e\n" % ("OW", atype, sigma, epsilon)

    return input_parm, gromacs_nonbond



#----------------------------------------------
def get_NONBONDED_PARM_INDEX(input_parm, atype_i, atype_j):
    """
    Get NONBONDED_PARM_INDEX for atomtype_i and atomtype_j LJ interaction.
    Returns and index using Amber prmtop file convention (start index = 1).
    If parm_data is modified as Python list - the index needs to be reduced by 1.
    """
    
    # get ATOM_TYPE_INDEX for both i and j
    i, j = input_parm.LJ_types[atype_i], input_parm.LJ_types[atype_j]
    # max is equal to total value of types (indexed starting with 1)
    #NTYPES = np.max(parm.LJ_types.values()) 
    ntypes = ntypes = parm.pointers['NTYPES'] 
    
    # NONBONDED_PARM_INDEX = NTYPES * [ ATOM_TYPE_INDEX(i) - 1 ] +  ATOM_TYPE_INDEX(j)
    #ij = parm.pointers['NTYPES'] * (i - 1) + j
    ij = parm.parm_data['NONBONDED_PARM_INDEX'][ntypes*(i-1)+j-1] - 1
    
    return ij


#----------------------------------------------
def remove_BCOEF(input_parm, atomtypes):
    """
    Remove BCOEF (C6) for the attractive part of LJ.
    Usage eg. to prevent ligand aggregation.
    ----
    input:
        input_parm - default system parameters
        atomtypes - list of atomtypes
    
    """
    for i in np.arange(len(atomtypes)):
        atype_i = atomtypes[i]
        for j in np.arange(i, len(atomtypes)):
            atype_j = atomtypes[j]
            
            # get cross interaction coefficient index
            ij = get_NONBONDED_PARM_INDEX(input_parm, atype_i, atype_j)
            
            # set BCOEF to zero!
            # adjust to pythonic start indexing at 0
            input_parm.parm_data["LENNARD_JONES_BCOEF"][ij - 1] = 0.0
            
            str_action = "Removing LJ attractive @%"+atype_i+"-@%"+atype_j+" pairwise interaction. (C6, BCOEF = 0.0)"
            log.write(str_action)
            if verbose: print(str_action)
    return


#----------------------------------------------
def add_nonbond_params(gmxtopin, gmx_nonbond, gmxtopout):
    """ adds lines in place of nonbond_params described in gmx_nonbond"""
    outfile=open(gmxtopout, "w")
    outfile.write(";\n;\tFile has been modified from default version of\n" +\
                  ";\t'%s'\n" % gmxtopin +\
                  ";\tusing Python script written by Vladas\n" +\
                  ";\tTime: %s\n" % time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) +\
                  ";\tChanges comprise of added [ nonbond_parms ] section\n")
    with open(gmxtopin, "r") as topol:
        atomtypes=0
        for line in topol.readlines():
            # only does something after 1st atomtypes
            if atomtypes==1:
                if line.startswith(";"):
                    continue
                elif line.startswith("\n") or line.startswith("["): # break
                    outfile.write("\n[ nonbond_params ]\n"+\
                                  "; type_i type_j  f.type   sigma   epsilon\n"+\
                                  "; f.type=1 means LJ (not Buckingham)\n"+\
                                  "; sigma&eps since mixing-rule = 2\n")
                    outfile.write(gmx_nonbond)
                    atomtypes+=1
            elif line.startswith("[ atomtypes ]"):
                atomtypes+=1
            outfile.write(line)
    outfile.close()
    print("written "+ gmxtopout)
    return


#####################################################
###################### MAIN #########################
#####################################################

# keep the time!
start_time = time.clock()

#######################
##### check the input
if not os.path.isfile(inputfile):
    print("ERROR: file %s not found, check your inputs" % inputfile)
    exit(1)    
else:
    input_parm = parmed.amber.readparm.AmberParm(inputfile, xyz=inputcoords)

print("%s\n S W I S H\n%s\n\n" % ("#"*42, "#"*42) +\
      "Input files read!\nStarting the script. This should take about a minute or two...\n")

if scale == []:
    sfactors = np.linspace(smin, smax, nreps)
else:
    sfactors = scale
    nreps = len(scale)

if 1.0 not in sfactors:
    print("WARNING: scaling factors does not include 1.0 - are you sure, it's OK?")
    print(sfactors)
    if not ignore:
        print("If above list looks OK, and you know what you are doing: restart with a flag '-ignore' to proceed.")
        exit(2)

######################
### begin the script

# define output file root!
if outputname == None:
    # use the same root as the input file with modifications to be added to the name!
    outputname = ".".join(inputfile.split(".")[:-1])

dirpath="./"

logfile = "SWISH_%s.log" % time.strftime("%Y-%m-%d", time.gmtime())    
print("Writing all the modifications to the log file: %s\n" % logfile)
log = open(logfile, 'w')
log.write("###\n# Sampling Water Interfaces though Scaled Hamiltonians\n# by Oleinikovas et al. (2016)\n###\n")
log.write("# This log was created on %s \n# by executing this command line:\n# %s\n###\n" %\
          (time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), ' '.join(sys.argv))) 


# add USER_COMMENTS flag to .prmtop file!
input_parm.add_flag("USER_COMMENTS", "(20a4)", data=None, num_items=0)

###########
#### HMR
if HMR:
    # perform HMR
    input_parm = do_HMR(input_parm, HM_Da)
    outputname += "+HMR"

if gmx:
    # save GRO base
    gmx_output = dirpath+outputname
    input_parm.save(gmx_output+".gro", overwrite=True)
    input_parm.save(gmx_output+".top", overwrite=True)


##############
### FRAGMENTS 
   
if frag_repel:
    # get atomtypes for non-H ligands atoms
    lig_atomtypes = get_ligand_atomtypes(input_parm, hydrogens=False)
    # remove posivitive LJ attraction term for all above seletcted
    #	remove_BCOEF(parm, atomtypes_lig)
    input_parm, lig_gmx_nonbond = add_interligand_repulsion(input_parm, lig_atomtypes, verbose=True)
    outputname += "+RepFrag"

    # SAVE INTERMEDIATE AMBER .PRMTOP
    input_parm.save(dirpath+'%s.prmtop' % (outputname), format="amber", overwrite=True)
    print_saved = '### New topology saved as %s.prmtop\n' % (outputname)
    log.write(print_saved)
    if verbose: print(print_saved)

    # SAVE INTERMEDIATE GMX .TOP
    if gmx:
        # GROMACS - modify nonbond
        gmx_nonbond = lig_gmx_nonbond
        # save
        add_nonbond_params(gmx_output+".top", gmx_nonbond, dirpath+outputname+".top")
	print_saved_gmx = '### New GMX topology saved as '+ dirpath+outputname+'.top\n'
	log.write(print_saved_gmx)
    	if verbose: print(print_saved_gmx)

else:
    lig_gmx_nonbond = ""



###############
### SWISH!

# prepare atomtypes for scaling!
if bind_pref:
    # create new atomtypes
    input_parm, atomtypes_SWISH = create_LJtypes(input_parm)
    if gmx:
	# need to renew the topology for different atomtypes!
    	input_parm.save(gmx_output+".top", overwrite=True)
else:
    # use all "apolar" atomtypes for scaling
    atomtypes_SWISH = get_apolar_atomtypes(input_parm)


for rep in np.arange(nreps):
    sfactor = sfactors[rep]
    
    print_replica = '### Replica %03d - scaling factor %.3f\n' % (rep, sfactor)
    log.write(print_replica)
    if verbose: print(print_replica)
    
    # apply SWISH-type scaling!
    input_parm_copy = copy.copy(input_parm)
    new_parm, SWISH_gmx_nonbond = scale_LJPairs(input_parm_copy, atomtypes_SWISH, sfactor, verbose=True)

    # check if off-diagonal terms have been changed
    if verbose: print("Off-diagonal terms changed: %s\n" % str(new_parm.has_NBFIX()))
       
    # SAVE AMBER .PRMTOP
    new_parm.save(dirpath+'%s+SWISH.%03d.prmtop' % (outputname, rep), format="amber", overwrite=True)
    print_saved = '### New topology saved as %s+SWISH.%03d.prmtop\n' % (outputname, rep)
    log.write(print_saved)
    if verbose: print(print_saved)

    # SAVE GMX .TOP
    if gmx:
        # GROMACS - modify nonbond
        gmx_nonbond = lig_gmx_nonbond + SWISH_gmx_nonbond
        # save
        add_nonbond_params(gmx_output+".top", gmx_nonbond, dirpath+outputname+"+SWISH%i.top" % rep)
	print_saved_gmx = '### New GMX topology saved as '+ dirpath+outputname+'+SWISH%i.top\n' % rep
	log.write(print_saved_gmx)
    	if verbose: print(print_saved_gmx)


# finish writing the log
log.close()

print("DONE!\nSee logfile for details: %s\n\nRunning time: %.2f s" % (logfile, time.clock() - start_time))
