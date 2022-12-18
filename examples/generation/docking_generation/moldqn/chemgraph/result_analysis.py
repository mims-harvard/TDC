import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

file_predix = "docking_smilesvaluelst_"
output_file_prefix = "moldqn_"
runs = [3, 11, 9]
points = [100, 500, 1000, 5000]

num_lst = [i * 100 for i in range(1, 51)]
for run_idx, run in enumerate(runs):
    file = file_predix + str(run) + ".pkl"
    smiles_score_lst = pickle.load(open(file, "rb"))
    smiles_score_lst = [(smiles, -score) for smiles, score in smiles_score_lst]
    print("run", run, "length", len(smiles_score_lst))

    top_1, top_10, top_100 = [], [], []
    for i in range(1, 51):
        smiles_value_lst = smiles_score_lst[: i * 100]
        smiles_value_lst.sort(key=lambda x: x[1])
        value_lst = [score for smiles, score in smiles_value_lst]
        top_1.append(value_lst[0])
        top_10.append(np.mean(value_lst[:10]))
        top_100.append(np.mean(value_lst[:100]))
    if run_idx == 0:
        plt.plot(num_lst, top_1, color="b", label="top-1")
        plt.plot(num_lst, top_10, color="r", label="top-10")
        plt.plot(num_lst, top_100, color="y", label="top-100")
    else:
        plt.plot(
            num_lst,
            top_1,
            color="b",
        )
        plt.plot(
            num_lst,
            top_10,
            color="r",
        )
        plt.plot(
            num_lst,
            top_100,
            color="y",
        )

    for point in points:
        smiles_value_lst = smiles_score_lst[:point]
        smiles_value_lst.sort(key=lambda x: x[1])
        smiles_value_lst = smiles_value_lst[:100]
        with open(
            output_file_prefix + str(run_idx) + "_" + str(point) + ".txt", "w"
        ) as fout:
            for smiles, value in smiles_value_lst:
                fout.write(smiles + "\t" + str(value) + "\n")


# num_lst = []
# num2dockingscore = {}
# file_lst = os.listdir(result_folder)
# for file in file_lst:
# 	file = os.path.join(result_folder, file)
# 	num = int(file.split('-')[1])
# 	num_lst.append(num)
# 	with open(file, 'r') as fin:
# 		lines = fin.readlines()
# 	smiles_score_lst = [(line.split()[0], float(line.split()[1])) for line in lines]
# 	score_lst = [i[1] for i in smiles_score_lst]
# 	score_lst.sort()
# 	assert score_lst[0] <= score_lst[1]
# 	num2dockingscore[num] = (score_lst[0], np.mean(score_lst[:10]), np.mean(score_lst))
# num_lst.sort()

# top_1 = [num2dockingscore[num][0] for num in num_lst]
# top_10 = [num2dockingscore[num][1] for num in num_lst]
# top_100 = [num2dockingscore[num][2] for num in num_lst]
# num_lst = [i for i in range(len(num_lst))]


# plt.plot(num_lst, top_1, color = 'b', label = 'top-1')
# plt.plot(num_lst, top_10, color = 'r', label = 'top-10')
# plt.plot(num_lst, top_100, color = 'y', label = 'top-100')
plt.legend()
plt.xlabel("# docking call")
plt.ylabel("docking score (DRD3) achieved by MolDQN")
plt.savefig("docking_iter.png")


"""
cd chemgraph 
scp -r tfu42@orcus1.cc.gatech.edu:/project/molecular_data/graphnn/mol_dqn_docking/chemgraph/result .   

"""
