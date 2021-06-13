'''

local

ls /Users/futianfan/Downloads/spring2021/TDC/tdc/test_docking.py 

cd /Users/futianfan/Downloads/spring2021/TDC/tdc

scp /Users/futianfan/Downloads/spring2021/TDC/tdc/test_docking.py  tfu42@orcus1.cc.gatech.edu:/project/molecular_data/graphnn/pyscreener

cd -

server 

screen -S docking_zinc_drd3 

source activate tdc 
export PATH=$PATH:/project/molecular_data/graphnn/docking/ADFRsuite_installed_directory/bin
export PATH=$PATH:/project/molecular_data/graphnn/docking/autodock_vina_1_1_2_linux_x86/bin 
cd /project/molecular_data/graphnn/pyscreener 


python test_docking.py 

'''
import random 
from tqdm import tqdm 
random.seed(8)
from tdc import Oracle 
from tdc.generation import MolGen
data = MolGen(name = 'ZINC')
df = data.get_data()['smiles'].to_list()
qed = Oracle(name = 'qed')

## 250000 
start_idx, end_idx = 200000, 250000 
batch_size = 20
df = df[start_idx:end_idx]



oracle = Oracle(name = 'Docking_Score', software='vina', 
                pyscreener_path = '/project/molecular_data/graphnn/pyscreener', 
         		receptors=['/project/molecular_data/graphnn/pyscreener/testing_inputs/DRD3.pdb'], 
         		center=(9, 22.5, 26), size=(15, 15, 15),
                buffer=10, path='/project/molecular_data/graphnn/pyscreener/my_test/', num_worker=3, ncpu=8)

'''

  1479 docking_zinc_drd3_0_50000
  1155 docking_zinc_drd3_100000_150000
  1227 docking_zinc_drd3_50000_100000

orcus1 		
	screen -X -S  44219.docking_zinc_drd3 kill
	cat /proc/cpuinfo | grep cores | uniq 
			14 cores
	num_worker = 3,  ncpu = 6, batch_size = 20
	2370--50000




code change
		modify start_idx, end_idx 
		num_worker, ncpu 
	2021/03/22 1:50 pm 



orcus2 
	cat /proc/cpuinfo | grep cores | uniq 
			14 cores
	num_worker = 3,  ncpu = 10, batch_size = 20
	2021/03/22 01:37 pm 

haum 
	cat /proc/cpuinfo | grep cores | uniq 
			16 cores 
	num_worker = 3,  ncpu = 6, batch_size = 20
	102543 -- 150000 			
	2021/03/22 01:45 pm 

uranus 
	cat /proc/cpuinfo | grep cores | uniq 
			14 cores 
	num_worker = 3,  ncpu = 6, batch_size = 20

eris 
	cat /proc/cpuinfo | grep cores | uniq 
		16 cores 
	num_worker = 3,  ncpu = 8, batch_size = 20



	2021/03/22 02:38 pm ----10k



'''
# oracle = Oracle(name = 'Docking_Score', software='vina', 
#                 pyscreener_path = '/project/molecular_data/graphnn/pyscreener', 
#          		receptors=['/project/molecular_data/graphnn/pyscreener/testing_inputs/5WIU.pdb'], 
#                 docked_ligand_file='/project/molecular_data/graphnn/pyscreener/testing_inputs/5WIU_with_ligand.pdb',
#                 buffer=10, path='/project/molecular_data/graphnn/pyscreener/my_test/', num_worker=4, ncpu=20)

# /project/molecular_data/graphnn/pyscreener/testing_inputs/5WIU.pdb
# /project/molecular_data/graphnn/pyscreener/testing_inputs/DRD3.pdb

output_file = "docking_zinc_drd3_" + str(start_idx) + '_' + str(end_idx) 

cnt = 0 
num_batch = int((end_idx - start_idx)/batch_size)
for i in tqdm(range(num_batch)):
	begin_idx = i * batch_size 
	end_idx =  (i+1) * batch_size  
	smiles_lst = df[begin_idx:end_idx]
	try:
		print(smiles_lst)
		smiles_lst = list(filter(lambda s:qed(s) > 0.2, smiles_lst))
		if smiles_lst == []:
			continue		
		score_lst = oracle(smiles_lst)
	except:
		continue 
	with open(output_file, 'a') as fout:
		for smiles, score in zip(smiles_lst, score_lst):
			fout.write(smiles + '\t' + str(score) + '\n')
			cnt += 1 
			print("validity ratio is", str(cnt / end_idx))








'''
cat docking_chembl_5WIU.results* > tmp 


'''

