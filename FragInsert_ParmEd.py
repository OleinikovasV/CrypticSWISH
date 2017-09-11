#!/home/vladas/software/anaconda/bin/python
d = """
===============================================================================
Replace waters with a specified fragment at a specified concentration.
Needs fragment topology (.prmtop , .rst7)

In addition, now allows switching modifying inter ligand interactions to prevent
aggregation (tuned repulsive LJ between ligand heavy atoms).
Developed for Amber type of protein force-fields. Assumes ligand atomtypes
are in lower cases (GAFF format).
------------------------
Example:
./ParmEd_insert_fragments.py -f <prmtop_file> -xyz <system_coords>
			     -frag BEN -conc 0.5 -lig_repel

The script relies on ParmEd and Numpy tools.

Vladas Oleinikovas
uccavol@ucl.ac.uk

version beta.3                                                    11/09/2017
================================================================================
"""


import numpy as np
import parmed
import os
import sys
import time
import copy
import argparse


#==========================================================================
# define functions
#==========================================================================

def calc_box_volume(input_parm):
    """ calculates periodic box volume for general triclinic box """
    # get side lengths and angles
    a, b, c = input_parm.box[:3]
    alpha, beta, gamma = input_parm.box[3:] * np.pi / 180 # convert to radians
    # calc volume
    V = a * b * c * \
        np.sqrt(1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 \
                + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma))
    return V

#--------------------------------------------------------------------------

def random3Drotation():
    """
    Returns random 3D rotation matrix about the origin.
    Use np.matmul(R_xyz, vector.T).T to rotate your 3D vectors.
    """
    # random uniform 3 angle distribution
    alpha, beta, gamma = np.random.uniform(low=-np.pi, high=np.pi, size=3)
    # R_x(theta) = [ 1 0 0 , 0 cos(theta) -sin(theta) , 0 sin(theta) cos(theta) ]
    # R_y(theta) = [ cos(theta) 0 sin(theta) , 0 1 0 , -sin(theta) 0 cos(theta) ]
    # R_z(theta) = [ cos(theta) -sin(theta) 0 , sin(theta) cos(theta) 0 , 0 0 1 ]
    R_x = np.array([[1, 0, 0], [0, np.cos(alpha), -np.sin(alpha)], [0, np.sin(alpha), np.cos(alpha)]])
    R_y = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
    R_z = np.array([[np.cos(gamma), -np.sin(gamma), 0], [np.sin(gamma), np.cos(gamma), 0], [0, 0, 1]])
    # matrix multiplication
    R_xyz = np.matmul(np.matmul(R_x, R_y), R_z)
    return R_xyz

#--------------------------------------------------------------------------

def selectReplaceableWaters(input_parm, cutoff=6.0):
    """
    Pick water residues that could be replaced, based on cutoff to non water residues and that
    has a full solvation shell (to avoid the edge!).
    Returns an array of 3D postions corresponding to water (OW) postions that are candidates
    to be replaced by fragments; and full water parameters to later remove possible clashes.
    """
    # get all waters
    Water = input_parm[":WAT"]
    # oxygen (OW) atom positions
    Coords = Water["@O"].coordinates

    ###! does not work as deepcopy is needed for coordinates, but shallow - for pointers
    ###! make a shallow bitwise copy of the initial parameters (no coordinate info!)
    ##! new_parm = copy.copy(input_parm)

    # only consider replacing Waters not too close to protein (4.0 A)
    # and sufficiently far from ions or other non-protein molecules to avoid clash (same as protein)
    notWater = input_parm["!:WAT"]
    notWaterCoords = notWater.coordinates

    ###! ... and pick not too far from protein (to avoid the box edge) (2.0 A from box solvation buffer should be enough!)
    ##! availcenters = np.array([(4.0 < np.min(np.linalg.norm(notWaterCoords - posOW, axis=1)) < 8.0) for posOW in Coords])
    availcenters1 = np.array([(np.min(np.linalg.norm(notWaterCoords - posOW, axis=1)) > cutoff) for posOW in Coords])
    # estimate water coordination within the first solvation shell (N.B. bulk has ~5, here > 5 since self included in count)
    availcenters2 = [(np.sum((np.linalg.norm(Coords - posOW, axis=1)) < 3.5) > 5 ) for posOW in Coords]
    # combine - both away enough from other molecules and with at least one solvation layer!
    availcenters = np.logical_and(availcenters1, availcenters2)

    # define available center positions for fragment insertion!
    centers = Coords[availcenters]
    
    return centers, Water

#--------------------------------------------------------------------------

def insertFragments(input_parm, fragment_parm, centers, N_frags, cutoff=6.0):
    """
    Insert randomly rotated fragments in place of centers provided picked at random.
    The available center positions are updated every step based on fragment closeness cutoff.
    Note, overlapping waters are retained at this step, just insertion!
    Returns inserted fragment parameters.
    Input_parm is updated implicitly!
    """
    # to monitor monitor progress
    progressbar = '.' * 20  # strings
    
    for n in np.arange(N_frags):

        sys.stdout.write('\r%s| Inserting fragments: %3d / %3d !' % (progressbar, n+1, N_frags))

        # deepcopy (contains coordinates) of the fragment to be inserted
        frag = copy.deepcopy(fragment)

        # get center of geometry (COG) of fragment and make it an origin
        CoG =  np.average(frag.coordinates, axis=0)
        frag.coordinates -= CoG
        # randomly rotate around X, Y and Z centered at fragment's COG
        frag.coordinates = np.matmul(random3Drotation(),frag.coordinates.T).T

        # pick at random water molecule to shift fragment center to its OW atom position
        frag.coordinates += centers[np.random.choice(np.arange(len(centers)))]

        # append this fragment to the new parm object
        input_parm += frag

        ### UPDATE iteratively available centers for fragment insertion!
        InsertedFrags = parm[":%s" % fragResName]
        # get full atom coords for already added fragments
        FragCoords = InsertedFrags.coordinates
        availcenters = np.array([(np.min(np.linalg.norm(FragCoords - posOW, axis=1)) > cutoff) for posOW in centers])
        if (sum(availcenters) == 0 and (n+1) != N_frags):
            # out of available positions to insert but not the last fragment!
            sys.exit("\nNo new available centers to pack more fragments (%d/%d)!" % (n, N_frags) +\
                     "\nStart reducing cutoff = %s ?\nOr define more optimal packing than random." % cutoff)
            # Could set up iterative scheme for which -RW would be decreasing gradually
            # could also set up a grid for more proper filling!
            # AddToBox: http://ambermd.org/tutorials/advanced/tutorial13/Solvation.html
        else:
            centers = centers[availcenters]

        if n % int(N_frags / 20)  == 0:
            # update progress bar !
            step = int((n+1) / N_frags * 20)
            progressbar = progressbar[:step] + '#' + progressbar[step+1:]
            sys.stdout.flush()
            
    return InsertedFrags

#--------------------------------------------------------------------------

def stripWaterClash(input_parm, Water, InsertedFrags, closeness=0.75):
    """
    Strips away water molecules that overlap with fragments. The criterion for striping away an overlapping
    WATER RESIDUEs is if the distance between any WATER ATOM and its nearest FRAGMENT ATOM is less than
    the sum of the two ATOMs' van der Waals radii multiplied by closeness.
    Typically removes ~ 15 waters per benzene if closeness = 1.0 and ~ 9 if 0.75 or ~ 12.5 if 2**(-1.0/6).
    Input_parm is updated implicitly.
    """
    # get full atom coordinates
    WaterCoords = Water.coordinates
    FragCoords = InsertedFrags.coordinates
    # get array with fragment Rmin/2
    FragRmin = np.array([atom.rmin for atom in InsertedFrags.atoms])
    
    # loop over water atoms to identify clashes with fragment atoms
    # get indices to be used in AMBER Selection mask!
    # number (the same in original input_parm) - not index (corresponds to slice)!
    # N.B. zero indexing vs starting 1 index in AMBER! (thus ":%s" % res.number+1)
    # list(set(...)) to remove duplicates!
    stripAmberRes = list(set( [str(atom.residue.number+1) for atom in Water.atoms\
                               if any(np.linalg.norm(FragCoords - WaterCoords[atom.idx], axis=1) <\
                                     (FragRmin + atom.rmin) * closeness)] ))
    # strips away residues using AMBER selection mask!
    input_parm.strip(":" + ",".join(stripAmberRes))
    print("\nRemoved %d overlaping waters!" %  len(stripAmberRes))

    return

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

#--------------------------------------------------------------------------
def add_interligand_repulsion(input_parm, lig_atomtypes, verbose=False):
    """
    Set interligang LJPair into an effectively repulsive potential
    excluding them from being within the same solvation shell
    (default setting: soft wall ~5A, decaying to 0 at sigma).
    Returns an updated parmed parameter object.
    """

    # GROMACS
    sigma = 0.8 #1.0 # nm
    epsilon = 1.5e-3    #1e-4 # kJ/mol
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


#==========================================================================
# Main script
#==========================================================================

########################################
######### DEFINE INPUTS ################
########################################
parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=d, epilog=" ")

parser.add_argument("-f", type=str, default='system.wat.leap.prmtop', help='input AMBER topology file name (default: %(default)s)')
parser.add_argument("-o", type=str, default=None, help='output AMBER topology file name (default: system+FRG+RepFrag.prmtop)')
parser.add_argument("-frag", type=str, default='BEN', help='input fragment name (default: %(default)s)')
parser.add_argument("-fraglibdir", type=str, default='./', help='path to directory of fragment library (.prmtop, .rst7) (default: %(default)s)')
parser.add_argument("-conc", type=float, default='0.25', help='approx. molar concentration of the fragment (default: %(default)s)')
parser.add_argument("-frag_repel", default=False, action='store_true', help="use inter ligand repulsion; (default: %(default)s)")
parser.add_argument("-v", "--verbose", action='store_true', help="be verbose")
parser.add_argument("-xyz", type=str, default='system.wat.leap.rst7', help='input AMBER coordinate file name (default: %(default)s)')
parser.add_argument("-gmx", default=False, action='store_true', help="save GROMACS topology, too (default: %(default)s)")


args = parser.parse_args()

inputfile = args.f
outputname = args.o
fraglibdir = args.fraglibdir
frag = args.frag
conc = args.conc
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


print("Starting script!\n")
# set timer!
start = time.time()

#### 0. load input !
dirpath = "./"

fraginputfile = frag + ".prmtop"
fraginputcoord = frag + ".rst7"

# load protein system solvated and neutralized with ions!
parm = parmed.load_file(inputfile, xyz=inputcoords)

# load fragment to insert (1 residue!)
fragment = parmed.load_file(fraglibdir+fraginputfile, xyz=fraglibdir+fraginputcoords)
fragResName = fragment.residues[0].name
print("Fragment %s is named %s in the topology!" % (frag, fragResName))

# calculate number of fragments based on concentration
volume = calc_box_volume(parm) # in A**3
N_frags = round(6.022e23 * volume * 1.0e-27 * conc) # rounded

# define output name
# removes the ending indicating file type (eg. .prmtop)
filename = dirpath + ".".join(inputfile.split(".")[:-1])
outputname = filename + "_%.2fM-%s" % (conc, frag)
# also save GROMACS parameter files (*.top, *.gro)!
gmx = True


# allowed closeness (in A) between inserted fragment center and:
#     1) protein (+ other nonwater residues) or
#     2) another previously inserted fragment
# roughly corresponds to 2nd solvation layer of water (~5.5A), thus should have at least one solvent layer between after insertion
cutoff = 6.0

print("Will attempt inserting %d %s fragments (%.2f mol/L)" % (N_frags, fragResName, (N_frags/ 6.022e23 / volume / 1.0e-27)))
print("This might take a few minutes ...")


########################################
######### INSERT FRAGMENTS #############
########################################

#### 1. pick water molecules positions that could be replaced (centers)
centers, Water = selectReplaceableWaters(parm, cutoff=cutoff)

#### 2. at random add suitable number of fragments in place of selected water oxygen atoms
InsertedFrags = insertFragments(parm, fragment, centers, N_frags, cutoff=cutoff)

# save intermediate files for inspection:
parm.save(outputname+"_raw.prmtop", overwrite=1)
parm.save(outputname+"_raw.rst7", overwrite=1)
if gmx:
    # save gromacs files, too!
    parm.save(outputname+"_raw.top", overwrite=1)
    parm.save(outputname+"_raw.gro", overwrite=1)
    # save gromacs fragment file to check too
    fragment.save(frag+".top", overwrite=1)
    fragment.save(frag+".gro", overwrite=1)

#### 3. once all the waters inserted - remove any waters overlapping with fragments
stripWaterClash(parm, Water, InsertedFrags, closeness=0.75)

#### 4. optionally add inter-fragment repulsion
if frag_repel:
    # get atomtypes for non-H ligands atoms
    lig_atomtypes = get_ligand_atomtypes(fragment, hydrogens=False)
    # remove posivitive LJ attraction term for all above seletcted
    parm, lig_gmx_nonbond = add_interligand_repulsion(parm, lig_atomtypes, verbose=True)
    outputname += "+RepFrag"


#### 4. save new topology and coordinates!
parm.save(outputname+".prmtop", overwrite=1)
parm.save(outputname+".rst7", overwrite=1)
if gmx:
    # save gromacs files, too! 
    parm.save(outputname+".top", overwrite=1)
    parm.save(outputname+".gro", overwrite=1)


#### 5. optionally add inter-fragment repulsion and save
if frag_repel:
    # get atomtypes for non-H ligands atoms
    lig_atomtypes = get_ligand_atomtypes(fragment, hydrogens=False)
    # remove posivitive LJ attraction term for all above seletcted
    parm, lig_gmx_nonbond = add_interligand_repulsion(parm, lig_atomtypes, verbose=True)
    gmxtopin = outputname # store gmx template topology name before updating
    outputname += "+RepFrag"

    parm.save(outputname+".prmtop", overwrite=1)
    parm.save(outputname+".rst7", overwrite=1)
    if gmx:
        # save gromacs files, too!
        parm.save(outputname+".top", overwrite=1)
        parm.save(outputname+".gro", overwrite=1)
        # save
        add_nonbond_params(gmxtopin+".top", gmx_nonbond, outputname+".top")
