
# rm -r results/smiles_lstm_hc_1 results/smiles_lstm_hc_2 results/smiles_lstm_hc_3 c
# scp -r tfu42@orcus1.cc.gatech.edu:/project/molecular_data/graphnn/pyscreener/smiles_lstm_hc/results.run.1 ./results/smiles_lstm_hc_1 
# scp -r tfu42@orcus1.cc.gatech.edu:/project/molecular_data/graphnn/pyscreener/smiles_lstm_hc/results.run.2 ./results/smiles_lstm_hc_2 
# scp -r tfu42@orcus1.cc.gatech.edu:/project/molecular_data/graphnn/pyscreener/smiles_lstm_hc/results.run.3 ./results/smiles_lstm_hc_3 

rm -r results/graph_ga_1 results/graph_ga_2 results/graph_ga_3 
scp -r tfu42@orcus1.cc.gatech.edu:/project/molecular_data/graphnn/pyscreener/graph_ga/results ./results/graph_ga_1  
scp -r tfu42@orcus1.cc.gatech.edu:/project/molecular_data/graphnn/pyscreener/graph_ga/results.2 ./results/graph_ga_2  
scp -r tfu42@orcus1.cc.gatech.edu:/project/molecular_data/graphnn/pyscreener/graph_ga/results.3 ./results/graph_ga_3  


