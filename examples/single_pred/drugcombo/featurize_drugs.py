# Create dataframe for drug features
# drug to fingerprint mapping
def get_fp(x):
    return list(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(x))) if Chem.MolFromSmiles(x) is not None else ''

df=pickle.load(open('./data/drugcomb_nci60.pkl', 'rb'))

fp_map = {}
all_drugs = set(df['Drug1'].unique()).union(set(df['Drug2'].unique()))
for drug in all_drugs:
    fp_map[drug] = [get_fp(drug)]
    
with open('./data/drug_features.pkl', 'wb') as handle:
    pickle.dump(fp_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
