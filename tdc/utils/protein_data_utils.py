"""
Class encapsulating protein data processing functions. Also supports running them in sequence.
Goal is to make it easier to integrate custom datasets not yet in TDC format.
"""

from Bio import Entrez, SeqIO
import mygene
from pandas import DataFrame
import requests

from .data_processing_utils import DataParser

class ProteinDataUtils(DataParser):
    """
    Class encapsulating protein data processing functions. Also supports running them in sequence.
    Goals are to make it easier to integrate custom datasets not yet in TDC format.
    Note: for running in sequence, this class inherits from data_processing_utils.DataParser
    """
    
    @classmethod
    def get_ncrna_sequence(cls, ncrna_id):
        """
        Fetches the nucleotide sequence for a given non-coding RNA ID from NCBI.

        Args:
        ncrna_id (str): The NCBI identifier for the non-coding RNA.

        Returns:
        str: The nucleotide sequence of the non-coding RNA, or a message if not found.
        """
        # Provide your email to NCBI to let them know who you are
        Entrez.email = "alejandro_velez-arce@hms.harvard.edu"

        try:
            # Fetch the sequence using efetch
            handle = Entrez.efetch(db="nucleotide", id=ncrna_id, rettype="fasta", retmode="text")
            record = SeqIO.read(handle, "fasta")
            handle.close()
            return str(record.seq)
        except Exception as e:
            return "Failed to retrieve sequence: " + str(e)
    
    @classmethod
    def get_amino_acid_sequence(cls, uniprot_id):
        """
        Fetches the amino acid sequence for a given UniProt ID.

        Args:
        uniprot_id (str): The UniProt ID for the protein.

        Returns:
        str: The amino acid sequence of the protein.
        """
        url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
        response = requests.get(url)

        if response.status_code == 200:
            fasta_content = response.text
            # Split the FASTA content into lines and discard the first line (header)
            sequence_lines = fasta_content.split('\n')[1:]
            # Join the lines to get the sequence
            sequence = ''.join(sequence_lines)
            return sequence
        else:
            return "Failed to retrieve sequence. Status code: " + str(response.status_code)
    
    @classmethod
    def get_protein_sequence(cls, gene_name : str) -> str:
        """
        Retrieves protein sequence for the given gene name.
        
        Args:
            gene_name (str): Gene name.
            
        Returns:
            str: Protein amino acid sequence.
        """
        assert isinstance(gene_name, str), (type(gene_name) , gene_name)
        mg = mygene.MyGeneInfo()
        # Query MyGene.info for the given gene name
        # You might need to adjust the fields based on the gene's specifics
        # 'fields': 'proteins' might vary depending on the data available for your gene
        gene_info = mg.query(gene_name, fields='all', species='human')
        try:
            # Attempt to extract the protein sequence
            # The path to the protein sequence might need adjustment based on the response structure
            if gene_name.upper() == "12CA5":
                return "YPYDVPDYA"  # hard-coded due to unavailability in mygene
            if gene_info['hits'][0]["type_of_gene"].lower() == "ncrna":
                ncbi_id = gene_info['hits'][0]['entrezgene']
                return cls.get_ncrna_sequence(ncbi_id)
                
            protid = gene_info['hits'][0]["uniprot"]["Swiss-Prot"]
            return cls.get_amino_acid_sequence(protid)
        except (IndexError, KeyError):
            # Handle cases where the protein sequence is not found
            return "Protein sequence not found for the given gene name."
    
    @classmethod
    def insert_protein_sequence(cls, gene_df: DataFrame, gene_column: str) -> DataFrame:
        """
        Inserts protein sequence for each gene in the given DataFrame.
        
        Args:
            gene_df (pd.DataFrame): Input DataFrame containing gene identifiers.
            gene_column (str): Column name in gene_df that contains gene identifiers.
            
        Returns:
            pd.DataFrame: DataFrame with an additional column for protein sequences.
        """
        # Check if gene_column exists in gene_df
        if gene_column not in gene_df.columns:
            raise ValueError(f"{gene_column} does not exist in the DataFrame.")

        # Ensure the DataFrame index is aligned
        # gene_df = gene_df.reset_index(drop=True)
        
        # Retrieve protein sequences for each gene and store them in a new column
        new_col = gene_df[gene_column].apply(cls.get_protein_sequence).tolist()
        assert len(new_col) == len(gene_df[gene_column]), (new_col, gene_df[gene_column])
        gene_df['sequence'] = new_col
        
        return gene_df