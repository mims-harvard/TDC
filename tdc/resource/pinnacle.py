from ..utils import general_load
from ..utils.load import resource_dataset_load 

class PINNACLE:
    
    def __init__(self, path="./data"):
        self.ppi_name = "pinnacle_global_ppi_edgelist"
        self.cell_tissue_mg_name = "cell_tissue_mg_edgelist"
        self.ppi = general_load(self.ppi_name, path, " ")
        self.ppi.columns = ["Protein A", "Protein B"]
        self.cell_tissue_mg = general_load(self.cell_tissue_mg_name, path, "\t")  # use tab as names were left with spaces
        self.cell_tissue_mg.columns = ["Tissue", "Cell"]
        self.embeds_name = "pinnacle_protein_embed"
        self.embeds = resource_dataset_load(self.embeds_name, path, [self.embeds_name])
    
    def get_ppi(self):
        return self.ppi
    
    def get_mg(self):
        return self.cell_tissue_mg
    
    def get_embeds(self):
        return self.embeds
   
   
    
    
    
    