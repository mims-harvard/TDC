from random import shuffle

for i in range(1, 4):
    file_1 = "results/graph_ga_" + str(i) + "_100.txt"
    file_2 = "results/graph_ga_" + str(i) + "_200.txt"

    with open(file_1, "r") as fin:
        lines = fin.readlines()
    smiles2value = {line.split()[0]: float(line.strip().split()[1]) for line in lines}
    with open(file_2, "r") as fin:
        lines = fin.readlines()
        shuffle(lines)
    s2v2 = {line.split()[0]: float(line.strip().split()[1]) for line in lines}
    for s, v in s2v2.items():
        if s not in smiles2value:
            smiles2value[s] = v
            break
    smiles_value_lst = [(smiles, value) for smiles, value in smiles2value.items()]
    smiles_value_lst.sort(key=lambda x: x[1])
    with open(file_1, "w") as fout:
        for smiles, value in smiles_value_lst:
            fout.write(smiles + "\t" + str(value) + "\n")
