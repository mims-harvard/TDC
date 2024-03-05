import sys
from vina import Vina
from time import time

ligand_pdbqt_file = sys.argv[1]
receptor_pdbqt_file = sys.argv[2]
output_file = sys.argv[3]
center = [sys.argv[4], sys.argv[5], sys.argv[6]]
center = [float(i) for i in center]
box_size = [sys.argv[7], sys.argv[8], sys.argv[9]]
box_size = [float(i) for i in box_size]


# print(ligand_pdbqt_file, receptor_pdbqt_file, output_file, center, box_size)
def docking(ligand_pdbqt_file, receptor_pdbqt_file, output_file, center,
            box_size):
    t1 = time()
    v = Vina(sf_name="vina")
    v.set_receptor(rigid_pdbqt_filename=receptor_pdbqt_file)
    v.set_ligand_from_file(ligand_pdbqt_file)
    v.compute_vina_maps(center=center, box_size=box_size)
    energy = v.score()
    energy_minimized = v.optimize()
    t2 = time()
    print("vina takes seconds: ", str(t2 - t1)[:5])
    with open(output_file, "w") as fout:
        fout.write(str(energy_minimized[0]))


docking(ligand_pdbqt_file, receptor_pdbqt_file, output_file, center, box_size)
"""
Example: 
    python XXXX.py data/1iep_ligand.pdbqt ./data/1iep_receptor.pdbqt ./data/out 15.190 53.903 16.917 20 20 20 
"""
