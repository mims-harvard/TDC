import os 
import numpy as np 
from vina import Vina
from time import time 
from rdkit import Chem
from random import random
def pdbfile2mol3d(pdbfile):
    return Chem.rdmolfiles.MolFromPDBFile(pdbfile, removeHs=False) 
    # return Chem.rdmolfiles.MolFromPDBFile(pdbfile, removeHs=False) 
    #### H is removed? 

def xyzfile2pdbfile(xyzfile, pdbfile):
    os.system("obabel -ixyz "+xyzfile+" -opdb -O "+pdbfile + " > /dev/null 2>&1") 


def xyzfile2mol3d(xyzfile):
    pdbfile = "data/" + str(random()) + '.pdb'
    xyzfile2pdbfile(xyzfile, pdbfile)
    mol3d = pdbfile2mol3d(pdbfile) 
    os.system("rm -f " + pdbfile)
    return mol3d


target_folder = "./receptors/"
pdbid_lst = ['1iep', '2rgp', '3eml', '3ny8', '4rlu', '4unn', '5mo4', '7l11']
for pdbid in pdbid_lst:
	print("-------", pdbid, "-------")
	#### docking center 
	v = Vina(sf_name='vina')
	pdbfile = target_folder + pdbid + "/" + pdbid + "_ligand.pdb" 
	mol3d = pdbfile2mol3d(pdbfile)
	conform = mol3d.GetConformer().GetPositions() 
	center = np.mean(conform, 0).tolist()

	receptor_pdbqt = target_folder + pdbid + "/" + pdbid + "_receptor.pdbqt"
	v.set_receptor(rigid_pdbqt_filename=receptor_pdbqt)

	ligand_pdbqt = target_folder + pdbid + "/" + pdbid + "_ligand.pdbqt"
	v.set_ligand_from_file(ligand_pdbqt)

	v.compute_vina_maps(center=center, box_size=[40, 40, 40])

	energy = v.score()
	print('Score before minimization: %.3f (kcal/mol)' % energy[0])

	energy_minimized = v.optimize()
	print('Score after minimization : %.3f (kcal/mol)' % energy_minimized[0])


