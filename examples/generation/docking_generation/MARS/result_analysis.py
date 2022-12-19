import os
import numpy as np
import matplotlib.pyplot as plt


for i in range(1, 4):
    result_folder = "result." + str(i) + ".run"
    file_lst = os.listdir(result_folder)
    num_lst = []
    num2dockingscore = {}
    for file in file_lst:
        if file[:10] == "best_after":
            num = int(file.split("_")[2])
        else:
            num = int(file.split(".")[0])
        if num > 5000:
            continue
        num_lst.append(num)
        file = os.path.join(result_folder, file)
        with open(file, "r") as fin:
            lines = fin.readlines()
        smiles_score_lst = [(line.split()[0], float(line.split()[1])) for line in lines]
        score_lst = [i[1] for i in smiles_score_lst]
        num2dockingscore[num] = (
            score_lst[0],
            np.mean(score_lst[:10]),
            np.mean(score_lst),
        )
    num_lst.sort()
    top_1 = [num2dockingscore[num][0] for num in num_lst]
    top_10 = [num2dockingscore[num][1] for num in num_lst]
    top_100 = [num2dockingscore[num][2] for num in num_lst]

    if i == 1:
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


plt.legend()
plt.xlabel("# docking call")
plt.ylabel("docking score (DRD3) achieved by MARS")
plt.savefig("docking_iter.png")

exit()


# result_folder = "result"
# num_lst = []
# num2dockingscore = {}
# file_lst = os.listdir(result_folder)
# for file in file_lst:
# 	file = os.path.join(result_folder, file)
# 	num = int(file.split('_')[2])
# 	num_lst.append(num)
# 	with open(file, 'r') as fin:
# 		lines = fin.readlines()
# 	smiles_score_lst = [(line.split()[0], float(line.split()[1])) for line in lines]
# 	score_lst = [i[1] for i in smiles_score_lst]
# 	num2dockingscore[num] = (score_lst[0], np.mean(score_lst[:10]), np.mean(score_lst))
# num_lst.sort()
# top_1 = [num2dockingscore[num][0] for num in num_lst]
# top_10 = [num2dockingscore[num][1] for num in num_lst]
# top_100 = [num2dockingscore[num][2] for num in num_lst]
