import os
import numpy as np
import matplotlib.pyplot as plt

result_folder = "result"
num_lst = []
num2dockingscore = {}
file_lst = os.listdir(result_folder)
for file in file_lst:
    file = os.path.join(result_folder, file)
    num = int(file.split("-")[1])
    num_lst.append(num)
    with open(file, "r") as fin:
        lines = fin.readlines()
    smiles_score_lst = [(line.split()[0], float(line.split()[1])) for line in lines]
    score_lst = [i[1] for i in smiles_score_lst]
    score_lst.sort()
    assert score_lst[0] <= score_lst[1]
    num2dockingscore[num] = (score_lst[0], np.mean(score_lst[:10]), np.mean(score_lst))
num_lst.sort()

top_1 = [num2dockingscore[num][0] for num in num_lst]
top_10 = [num2dockingscore[num][1] for num in num_lst]
top_100 = [num2dockingscore[num][2] for num in num_lst]
num_lst = [i for i in range(len(num_lst))]
num_lst = [i / max(num_lst) * 5000 for i in num_lst]


plt.plot(num_lst, top_1, color="b", label="top-1")
plt.plot(num_lst, top_10, color="r", label="top-10")
plt.plot(num_lst, top_100, color="y", label="top-100")
plt.legend()
plt.xlabel("# docking call")
plt.ylabel("docking score (DRD3) achieved by MolDQN")
plt.savefig("docking_iter.png")


"""
cd chemgraph 
scp -r tfu42@orcus1.cc.gatech.edu:/project/molecular_data/graphnn/mol_dqn_docking/chemgraph/result .   

"""
