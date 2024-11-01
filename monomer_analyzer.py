#!/usr/bin/env python3

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, DataStructs, PandasTools, Crippen, AllChem
import os, sys, requests, tqdm, re, argparse
from collections import defaultdict
import numpy as np
from rdkit.Contrib.IFG import ifg
from tqdm import tqdm
import xml.etree.ElementTree as ET

# Add SA_Score directory to the system path to import sascorer
sys.path.append(os.path.abspath("/Users/yunaoikawa/src/pep_NNAA2/SA_Score"))
import sascorer

sys.path.append(os.path.abspath("/Users/yunaoikawa/src/pep_NNAA2/NP_Score"))
import npscorer
# Load the natural product scoring model
fscore = npscorer.readNPModel()

# argparse
parser = argparse.ArgumentParser(description='Analyse non-natural amino acids (NNAA) from PubChem.')
parser.add_argument('--input_file', help='Input file containing the NNAA data in TSV or TXT format.', default='NNAA.txt')
parser.add_argument('--mols_file', help='Input the mol data TSV or TXT format if you want to rewrite the count column.', default='mols.txt')
parser.add_argument('-fetch_names', help='Fetch the names of the NNAA from PubChem.', action='store_true')
parser.add_argument('--target_type', help='ncAAs or peptides?', default='ncAAs')
args = parser.parse_args()

# Load your data into a DataFrame
df = pd.read_csv(args.input_file, sep='\t')

def add_canonical_smiles(df):
    # List of canonical SMILES strings
    canonical_smiles_list = [
        "C1=CC=C2C(=C1)C(=CN2)C[C@@H](C(=O)O)N",  # Trytophan (W)
        "C(C[C@@H](C(=O)O)N)CNC(=N)N",             # Arginine (R)
        "C1=C(NC=N1)C[C@@H](C(=O)O)N",             # Histidine (H)
        "C1C[C@H](NC1)C(=O)O",                     # Proline (P)
        "C(CCN)C[C@@H](C(=O)O)N",                  # Lysine (K)
        "CSCC[C@@H](C(=O)O)N",                     # Methionine (M)
        "C(CC(=O)N)[C@@H](C(=O)O)N",               # Asparagine (N)
        "C([C@@H](C(=O)O)N)C(=O)N",                # Glutamine (Q)
        "C(CC(=O)O)[C@@H](C(=O)O)N",               # Glutamic acid (E)
        "OC(=O)C[C@@H](C(=O)O)N",                  # Aspartic acid (D)
        "C1=CC(=CC=C1C[C@@H](C(=O)O)N)O",          # Tyrosine (Y)
        "C1=CC=C(C=C1)C[C@@H](C(=O)O)N",           # Phenylalanine (F)
        "CC[C@H](C)[C@@H](C(=O)O)N",               # Valine (V)
        "CC(C)C[C@@H](C(=O)O)N",                   # Leucine (L)
        "CC(C)[C@@H](C(=O)O)N",                    # Isoleucine (I)
        "C[C@H]([C@@H](C(=O)O)N)O",                # Threonine (T)
        "C([C@@H](C(=O)O)N)S",                     # Cysteine (C)
        "C([C@@H](C(=O)O)N)O",                     # Serine (S)
        "C[C@@H](C(=O)O)N",                        # Alanine (A)
        "C(C(=O)O)N"                               # Glycine (G)
    ]

    # One-letter amino acid codes corresponding to the SMILES
    one_letter_codes = [
        'W', 'R', 'H', 'P', 'K', 'M', 'N', 'Q', 'E', 'D', 
        'Y', 'F', 'V', 'L', 'I', 'T', 'C', 'S', 'A', 'G'
    ]
    
    # Create a DataFrame for canonical SMILES
    canonical_df = pd.DataFrame({
        'ID': one_letter_codes,   # Add one-letter codes as IDs
        'SMILES': canonical_smiles_list,
        'CANONICAL': ['True'] * len(canonical_smiles_list),
        'TERMINAL': ['NotTer'] * len(canonical_smiles_list),
        'ROMol': [Chem.MolFromSmiles(smiles) for smiles in canonical_smiles_list]
    })
    
    # Concatenate the canonical DataFrame with the input DataFrame
    df = pd.concat([canonical_df, df], ignore_index=True)
    
    return df

def cal_tanimoto(mol):
    # Calculate the Tanimoto similarity between the molecule and L-Glycine
    l_glycine = Chem.MolFromSmiles("C(C(=O)O)N")
    fp1 = rdMolDescriptors.GetMorganFingerprint(mol, 2)
    fp2 = rdMolDescriptors.GetMorganFingerprint(l_glycine, 2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def find_primary_amine(mol):
    alpha_smarts = Chem.MolFromSmarts('[NH2][C](C(=O)O)')
    return str(mol.HasSubstructMatch(alpha_smarts))

def fetch_pubchem_name(NNAA_smiles):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{NNAA_smiles}/property/Title/JSON"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data['PropertyTable']['Properties'][0].get('Title', 'NULL')
    except (requests.exceptions.RequestException, KeyError, IndexError) as e:
        print(f"An error occurred with PubChem: {e}")
        return "NULL"

def fetch_chembl_similarity(smiles, similarity_threshold=100):
    try:
        url = f"https://www.ebi.ac.uk/chembl/api/data/similarity/{smiles}/{similarity_threshold}"
        response = requests.get(url)
        response.raise_for_status()

        # Parse the XML response
        root = ET.fromstring(response.content)
        chembl_ids = []
        
        # Extract the ChEMBL ID from each molecule
        for molecule in root.findall('.//molecule'):
            chembl_id = molecule.find('.//molecule_chembl_id')
            if chembl_id is not None:
                chembl_ids.append(chembl_id.text)

        # Return the ChEMBL IDs or 'NULL' if none found
        return chembl_ids if chembl_ids else ["NULL"]

    except requests.exceptions.RequestException as e:
        print(f"An error occurred with ChEMBL: {e}")
        return ["NULL"]

def fetch_names(smiles):
    pubchem_name = fetch_pubchem_name(smiles)
    chembl_names = str(fetch_chembl_similarity(smiles))
    return [pubchem_name] + [chembl_names]


def fetch_pubchem_property(props, label, name):
    for prop in props:
        if prop['urn']['label'] == label and prop['urn']['name'] == name:
            if 'fval' in prop['value']:
                return prop['value']['fval']
            elif 'sval' in prop['value']:
                return prop['value']['sval']
            elif 'ival' in prop['value']:
                return prop['value']['ival']
    return "NULL"

def count_monomers(mols_df):
    monomers_dict = defaultdict(int)  # Default to int for counting occurrences
    for sequence in mols_df['SEQUENCE']:
        if type(sequence) != float and len(sequence) > 0:
            # Split sequence into monomers (assumed to be uppercase letters followed by optional lowercase letters)
            tokens = re.findall('[A-Z][^A-Z]*', sequence)
            for token in tokens:
                monomers_dict[token] += 1  # Increment count for each monomer
    return monomers_dict

def identify_functional_groups(mol, NNAAcount=1):
    count = defaultdict(int)
    fgs = ifg.identify_functional_groups(mol)
    
    # Iterate over the identified functional groups
    for fg in fgs:
        string = str(fg)
        # Extract the functional group type
        fg_type = string.split("type='")[1].split("'")[0]
        count[fg_type] += NNAAcount
    
    return count


df = df.dropna(subset=['SMILES'])
df = df.drop_duplicates(subset=['SMILES'])
df['ROMol'] = df['SMILES'].apply(Chem.MolFromSmiles)

if args.target_type == 'ncAAs':
    df['CANONICAL'] = "False"
    df = add_canonical_smiles(df)
    df['TERMINAL'] = df['TERMINAL'].map({'ter': "True", 'NotTer': "False"})
    df['HAS PRIMARY AMINE'] = df['ROMol'].apply(find_primary_amine)
elif args.target_type == 'peptides':
    df = df.dropna(subset=['SEQUENCE'])
    df['CONTAINS NON-CANONICAL'] = df['SEQUENCE'].apply(lambda x: 'True' if 'X' in x else 'False')

if args.fetch_names:
    for smiles in tqdm(df['SMILES'], desc="Fetching names from PubChem and ChEMBL"):
        names = fetch_names(smiles)
        print(names)
        df.loc[df['SMILES'] == smiles, ['PUBCHEM', 'CHEMBL']] = names[:2]  # Storing PubChem and one ChEMBL name

df['Weight (g/mol)'] = df['ROMol'].apply(Chem.rdMolDescriptors.CalcExactMolWt)
df['ClogP'] = df['ROMol'].apply(Crippen.MolLogP)
df['Topological Polar Surface Area (Å)'] = df['ROMol'].apply(Chem.rdMolDescriptors.CalcTPSA)
df['Charge'] = df['ROMol'].apply(Chem.rdmolops.GetFormalCharge)
df['Number of Aromatic Rings'] = df['ROMol'].apply(Chem.rdMolDescriptors.CalcNumAromaticRings)
df['Number of Rotatable Bonds'] = df['ROMol'].apply(Chem.rdMolDescriptors.CalcNumRotatableBonds)
df['Number of Hydrogen Bond Acceptors'] = df['ROMol'].apply(Chem.rdMolDescriptors.CalcNumLipinskiHBA)
df['Number of Hydrogen Bond Donors'] = df['ROMol'].apply(Chem.rdMolDescriptors.CalcNumLipinskiHBD)
df['Tanimoto Similarity with L-Glycine'] = df['ROMol'].apply(cal_tanimoto)
df['Fraction sp3'] = df['ROMol'].apply(lambda mol: rdMolDescriptors.CalcFractionCSP3(mol))
df['Synthetic Accessibility Score'] = df['ROMol'].map(sascorer.calculateScore)
df['Natural Product Likeness'] = df['ROMol'].apply(lambda mol: npscorer.scoreMol(mol, fscore))
df['Functional Groups'] = df['ROMol'].apply(identify_functional_groups)

if args.mols_file:
    # Read the file and drop duplicates based on 'SEQUENCE'
    mols_df = pd.read_csv(args.mols_file, sep='\t')
    mols_df = mols_df.drop_duplicates(subset=['SEQUENCE'])
    
    # Create a dictionary of monomers
    monomers_dict = count_monomers(mols_df)
    
    # Use map to directly assign values from monomers_dict to 'COUNT' column based on 'ID'
    df['COUNT'] = df['ID'].map(monomers_dict).fillna(df['COUNT'])
    
    print('Count updated.')

# Replace NaN and INF with a specific value, e.g., empty string ''
df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace INF with NaN
df.fillna('', inplace=True)  # Replace NaN with an empty string

# sort by count
df = df.sort_values(by=['COUNT'], ascending=False)

try:
    PandasTools.SaveXlsxFromFrame(df, f"{args.input_file.split('.')[0]}_analysis.xls", molCol='ROMol', size=(150,150))
    print("XLS file saved successfully.")
except:
    print("Error saving the XLS file.")

# Save the updated DataFrame to a new CSV file
df.drop(columns=['ROMol'], inplace=True)

output_file = f"{args.input_file.split('.')[0]}_analysis.txt"
df.to_csv(output_file, sep='\t', index=False)

print("Calculations completed and saved to:", output_file)
