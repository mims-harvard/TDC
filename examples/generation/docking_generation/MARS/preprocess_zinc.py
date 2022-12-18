input_file = "data/zinc.tab"
output_file = "data/zinc.txt"

with open(input_file, "r") as fin, open(output_file, "w") as fout:
    lines = fin.readlines()
    lines = lines[1:]
    for line in lines:
        line = line.strip()
        line = line[1:-1]
        fout.write(line + "\n")
