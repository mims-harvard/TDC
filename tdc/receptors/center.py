import os 
import numpy as np 
from vina import Vina
v = Vina()
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


# box的center坐标可以直接计算native ligand的平均坐标得到：
target_folder = "./receptors/"
pdbid_lst = ['1iep', '2rgp', '3eml', '3ny8', '4rlu', '4unn', '5mo4', '7l11']
for pdbid in pdbid_lst:
	pdbfile = target_folder + pdbid + "/" + pdbid + "_ligand.pdb" 
	mol3d = pdbfile2mol3d(pdbfile)
	conform = mol3d.GetConformer().GetPositions() 
	center = np.mean(conform, 0).tolist()
	center = ' '.join([str(i) for i in center])
	filename = target_folder + pdbid + "/" + pdbid + "_center.txt" 
	with open(filename, 'w') as fout:
		fout.write(center)



