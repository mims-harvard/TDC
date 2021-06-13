import matplotlib.pyplot as plt
import json 
with open('tmp', 'r') as fin:
	lines = fin.readlines() 

for line in lines:
	score = line.strip().split()[1]
	if score == 'None':
		continue 
	score = float(score)
	# if score > 0:
	# 	print(line)


scores = [float(line.strip().split()[1]) for line in lines if line.strip().split()[1]!='None']
scores = [i for i in scores if i < 0]  ##### remove the nonnegative docking score 


guacamol_file = "goal_directed_docking300.json"


dic = json.load(open(guacamol_file, 'r'))
guacamol_scores = dic['results'][0]['optimized_molecules']
guacamol_scores = [-i[1] for i in guacamol_scores]
# print(guacamol_scores)

plt.hist(scores, bins = 50, density = True, label = 'chembl', alpha=0.5, color = 'b')
plt.hist(guacamol_scores, bins = 50, density = True, label = 'GA top-100', alpha=0.5, color = 'r')
plt.legend()

plt.savefig("docking_score_distribution.png")




'''

cd /Users/futianfan/Downloads/spring2021/TDC/tdc

cat docking* > tmp 

scp tfu42@orcus1.cc.gatech.edu:/project/molecular_data/graphnn/pyscreener/tmp .

python observe_docking_score_distribution.py

'''