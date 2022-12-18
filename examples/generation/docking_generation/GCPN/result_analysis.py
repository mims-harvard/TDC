import os
import numpy as np
import matplotlib.pyplot as plt

i_lst = [1, 2, 3]
run2result = {}
for i in i_lst:
    result_folder = "result." + str(i) + ".done"
    num_lst = []
    num2dockingscore = {}
    for nn in range(1, 51):
        num = nn * 100
        file = os.path.join(result_folder, str(num) + ".txt")
        with open(file, "r") as fin:
            lines = fin.readlines()
        smiles_score_lst = [(line.split()[0], float(line.split()[1])) for line in lines]
        score_lst = [i[1] for i in smiles_score_lst]
        num2dockingscore[num] = (
            score_lst[0],
            np.mean(score_lst[:10]),
            np.mean(score_lst),
        )
        num_lst.append(num)
    run2result[i] = num2dockingscore

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
plt.ylabel("docking score (DRD3) achieved by GCPN")
plt.savefig("gcpn_docking.png")


"""
scp -r tfu42@orcus1.cc.gatech.edu:/project/molecular_data/graphnn/GCPN/result .   

cp result.1.done/100.txt ../tdc_leaderboard/gcpn/gcpn_1_100.txt 
cp result.1.done/500.txt ../tdc_leaderboard/gcpn/gcpn_1_500.txt 
cp result.1.done/1000.txt ../tdc_leaderboard/gcpn/gcpn_1_1000.txt 
cp result.1.done/5000.txt ../tdc_leaderboard/gcpn/gcpn_1_5000.txt 

cp result.2.done/100.txt ../tdc_leaderboard/gcpn/gcpn_2_100.txt 
cp result.2.done/500.txt ../tdc_leaderboard/gcpn/gcpn_2_500.txt 
cp result.2.done/1000.txt ../tdc_leaderboard/gcpn/gcpn_2_1000.txt 
cp result.2.done/5000.txt ../tdc_leaderboard/gcpn/gcpn_2_5000.txt 

cp result.3.done/100.txt ../tdc_leaderboard/gcpn/gcpn_3_100.txt 
cp result.3.done/500.txt ../tdc_leaderboard/gcpn/gcpn_3_500.txt 
cp result.3.done/1000.txt ../tdc_leaderboard/gcpn/gcpn_3_1000.txt 
cp result.3.done/5000.txt ../tdc_leaderboard/gcpn/gcpn_3_5000.txt 


"""
