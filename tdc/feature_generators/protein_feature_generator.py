"""
Class encapsulating protein data processing functions. Also supports running them in sequence.
Goal is to make it easier to integrate custom datasets not yet in TDC format.
"""

from Bio import Entrez, SeqIO
import mygene
from pandas import DataFrame
import requests

from .data_feature_generator import DataFeatureGenerator


class ProteinFeatureGenerator(DataFeatureGenerator):
    """
    Class encapsulating protein data processing functions. Also supports running them in sequence.
    Goals are to make it easier to integrate custom datasets not yet in TDC format.
    Note: for running in sequence, this class inherits from data_processing_utils.DataFeatureGenerator
    """

    _SEQUENCE_MAP = {
        "ACE2": (
            "protein-coding",
            "MSSSSWLLLSLVAVTAAQSTIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQSTLAQMYPLQEIQNLTVKLQLQALQQNGSSVLSEDKSKRLNTILNTMSTIYSTGKVCNPDNPQECLLLEPGLNEIMANSLDYNERLWAWESWRSEVGKQLRPLYEEYVVLKNEMARANHYEDYGDYWRGDYEVNGVDGYDYSRGQLIEDVEHTFEEIKPLYEHLHAYVRAKLMNAYPSYISPIGCLPAHLLGDMWGRFWTNLYSLTVPFGQKPNIDVTDAMVDQAWDAQRIFKEAEKFFVSVGLPNMTQGFWENSMLTDPGNVQKAVCHPTAWDLGKGDFRILMCTKVTMDDFLTAHHEMGHIQYDMAYAAQPFLLRNGANEGFHEAVGEIMSLSAATPKHLKSIGLLSPDFQEDNETEINFLLKQALTIVGTLPFTYMLEKWRWMVFKGEIPKDQWMKKWWEMKREIVGVVEPVPHDETYCDPASLFHVSNDYSFIRYYTRTLYQFQFQEALCQAAKHEGPLHKCDISNSTEAGQKLFNMLRLGKSEPWTLALENVVGAKNMNVRPLLNYFEPLFTWLKDQNKNSFVGWSTDWSPYADQSIKVRISLKSALGDKAYEWNDNEMYLFRSSVAYAMRQYFLKVKNQMILFGEEDVRVANLKPRISFNFFVTAPKNVSDIIPRTEVEKAIRMSRSRINDAFRLNDNSLEFLGIQPTLGPPNQPPVSIWLIVFGVVMGVIVVGIVILIFTGIRDRKKKNKARSGENPYASIDISKGENNPGFQNTDDVQTSF"
        ),
        "12CA5": ("protein-coding", "YPYDVPDYA")
    }

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
            handle = Entrez.efetch(db="nucleotide",
                                   id=ncrna_id,
                                   rettype="fasta",
                                   retmode="text")
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
            return "Failed to retrieve sequence. Status code: " + str(
                response.status_code)

    @classmethod
    def get_protein_sequence(cls, gene_name: str) -> str:
        """
        Retrieves protein sequence for the given gene name.
        
        Args:
            gene_name (str): Gene name.
            
        Returns:
            str: Protein amino acid sequence.
        """
        assert isinstance(gene_name, str), (type(gene_name), gene_name)
        mg = mygene.MyGeneInfo()
        # Query MyGene.info for the given gene name
        # You might need to adjust the fields based on the gene's specifics
        # 'fields': 'proteins' might vary depending on the data available for your gene
        if gene_name in cls._SEQUENCE_MAP:  # genes for which mygene seems to be mislabled
            return cls._SEQUENCE_MAP[gene_name][1]
        gene_info = mg.query(gene_name, fields='all', species='human')
        try:
            # Attempt to extract the protein sequence
            # The path to the protein sequence might need adjustment based on the response structure
            if gene_info['hits'][0]["type_of_gene"].lower() == "ncrna":
                ncbi_id = gene_info['hits'][0]['entrezgene']
                return cls.get_ncrna_sequence(ncbi_id)

            protid = gene_info['hits'][0]["uniprot"]["Swiss-Prot"]
            return cls.get_amino_acid_sequence(protid)
        except (IndexError, KeyError):
            # Handle cases where the protein sequence is not found
            return "Protein sequence not found for the given gene name."

    @classmethod
    def get_type_of_gene(cls, gene_name: str) -> str:
        assert isinstance(gene_name, str), (type(gene_name), gene_name)
        mg = mygene.MyGeneInfo()
        # Query MyGene.info for the given gene name
        # You might need to adjust the fields based on the gene's specifics
        # 'fields': 'proteins' might vary depending on the data available for your gene
        if gene_name in cls._SEQUENCE_MAP:
            return cls._SEQUENCE_MAP[gene_name][0]
        gene_info = mg.query(gene_name, fields='all', species='human')
        try:
            return gene_info["hits"][0][
                "type_of_gene"] if gene_name != "12CA5" else "protein-coding"
        except:
            return "Could not find type of gene"

    @classmethod
    def insert_protein_sequence(cls, dataset: DataFrame,
                                gene_column: str) -> DataFrame:
        """
        Inserts protein sequence for each gene in the given DataFrame.
        
        Args:
            gene_df (pd.DataFrame): Input DataFrame containing gene identifiers.
            gene_column (str): Column name in gene_df that contains gene identifiers.
            
        Returns:
            pd.DataFrame: DataFrame with an additional column for protein sequences.
        """

        def helper(gene_name):
            if gene_name in memo:
                return memo[gene_name]
            memo[gene_name] = cls.get_protein_sequence(gene_name)
            return memo[gene_name]

        def helper_type(gene_name):
            if gene_name in memo_type:
                return memo_type[gene_name]
            memo_type[gene_name] = cls.get_type_of_gene(gene_name)
            return memo_type[gene_name]

        memo: dict = {}  # To store already computed values
        memo_type: dict = {}
        # Check if gene_column exists in gene_df
        if gene_column not in dataset.columns:
            raise ValueError(f"{gene_column} does not exist in the DataFrame.")

        # Ensure the DataFrame index is aligned
        # gene_df = gene_df.reset_index(drop=True)

        # Retrieve protein sequences for each gene and store them in a new column
        new_col = dataset[gene_column].apply(helper).tolist()
        assert len(new_col) == len(dataset[gene_column]), (new_col,
                                                           dataset[gene_column])
        dataset['protein_or_rna_sequence'] = new_col
        dataset["gene_type"] = dataset[gene_column].apply(helper_type).tolist()

        return dataset
