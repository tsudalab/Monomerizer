#!/usr/bin/env python3

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, DataStructs, PandasTools, Crippen, rdFreeSASA, AllChem
import requests, tqdm, re, argparse
import numpy as np

# argparse
parser = argparse.ArgumentParser(description='Analyse non-natural amino acids (NNAA) from PubChem.')
parser.add_argument('--input_file', help='Input file containing the mol data in TSV or TXT format.', default='mols.txt')
parser.add_argument('-fetch_names', help='Fetch the names of the NNAA from PubChem.', action='store_true')
parser.add_argument('-output_excel', help='Output the analysis to an Excel file.', action='store_true')
parser.add_argument('--ignore_short', help='How many amino acids are considered short?', type=int, default=0)
args = parser.parse_args()

# Load your data into a DataFrame
df = pd.read_csv(args.input_file, sep='\t')

def cal_tanimoto(mol):
    # Calculate the Tanimoto similarity between the molecule and L-Glycine
    l_glycine = Chem.MolFromSmiles("C(C(=O)O)N")
    fp1 = rdMolDescriptors.GetMorganFingerprint(mol, 2)
    fp2 = rdMolDescriptors.GetMorganFingerprint(l_glycine, 2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def find_primary_amine(mol):
    mol = Chem.AddHs(mol)
    alpha_smarts = Chem.MolFromSmarts('[NH2][C](C(=O)O)')
    return str(mol.HasSubstructMatch(alpha_smarts))

def fetch_pubchem_name(NNAA_smiles):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{NNAA_smiles}/property/Title/JSON"
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        data = response.json()
        return data['PropertyTable']['Properties'][0].get('Title', 'NULL')
    except (requests.exceptions.RequestException, KeyError, IndexError) as e:
        print(f"An error occurred: {e}")
        return "NULL"

def fetch_names_from_pubchem(smiles):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/JSON"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # Raises an HTTPError if the status is 4xx, 5xx
        response_json = response.json()
        
        # Extract properties from the JSON response
        properties_data = response_json['PC_Compounds'][0]
        props = properties_data['props']
        
        pubchem = fetch_pubchem_name(smiles)
        iupac = fetch_pubchem_property(props, "IUPAC Name", "Preferred")
        return [pubchem, iupac]

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return ["NULL"] * 2


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

def calculate_tpsa_ratio(mol):
    # Calculate the fraction of the topological polar surface area (TPSA)
    mol = Chem.AddHs(mol)
    try:
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol)  # Embed 3D conformer
            AllChem.UFFOptimizeMolecule(mol)  # Optimize the conformer with force field
        radii = rdFreeSASA.classifyAtoms(mol)
        return Chem.rdMolDescriptors.CalcTPSA(mol) / rdFreeSASA.CalcSASA(mol, radii)
    except ValueError:
        return np.nan
    
def rule_of_five(row):
    if (row['Weight (g/mol)'] <= 500 and
        row['Number of Hydrogen Bond Donors'] <= 5 and
        row['Number of Hydrogen Bond Acceptors'] <= 10 and
        row['ClogP'] <= 5):
        return True
    else:
        return False


if args.fetch_names:
    for tautomers in tqdm(df['SMILES'], desc="Fetching names from PubChem"):
        for smiles in tautomers.split(','):
            names = fetch_names_from_pubchem(smiles)
            if names[0] != "NULL":
                df.loc[df['SMILES'] == smiles, ['PubChem', 'IUPAC Name']] = names
                break
def count_capital_letters(seq):
    return sum(1 for c in seq if c.isupper())

# deduplicate
df.dropna(subset=['SEQUENCE'], inplace=True)
df.drop_duplicates(subset='SMILES', inplace=True)

if args.ignore_short:
    min_capital_letters = int(args.ignore_short)  # Ensure it's an integer
    df = df[df['SEQUENCE'].apply(count_capital_letters) >= min_capital_letters]

df['ROMol'] = df['SMILES'].apply(lambda x: Chem.MolFromSmiles(str(x), sanitize=True) if pd.notnull(x) and isinstance(x, str) and Chem.MolFromSmiles(str(x)) is not None else np.nan)
# Drop rows where ROMol is NaN (i.e., unsuccessful conversions)
df.dropna(subset=['ROMol'], inplace=True)

df['Amino Acid Length'] = df['SEQUENCE'].apply(count_capital_letters)
df['Weight (g/mol)'] = df['ROMol'].apply(Chem.rdMolDescriptors.CalcExactMolWt)
df['Number of bonds'] = df['ROMol'].apply(lambda mol: mol.GetNumBonds())
df['ClogP'] = df['ROMol'].apply(Crippen.MolLogP)
df['MolLogP'] = df['ROMol'].apply(Crippen.MolLogP)
df['Topological Polar Surface Area (Å)'] = df['ROMol'].apply(Chem.rdMolDescriptors.CalcTPSA)
df['Charge'] = df['ROMol'].apply(Chem.rdmolops.GetFormalCharge)
df['Number of Aromatic Rings'] = df['ROMol'].apply(Chem.rdMolDescriptors.CalcNumAromaticRings)
df['Number of Rotatable Bonds'] = df['ROMol'].apply(Chem.rdMolDescriptors.CalcNumRotatableBonds)
df['Number of Hydrogen Bond Acceptors'] = df['ROMol'].apply(Chem.rdMolDescriptors.CalcNumLipinskiHBA)
df['Number of Hydrogen Bond Donors'] = df['ROMol'].apply(Chem.rdMolDescriptors.CalcNumLipinskiHBD)
#df['Topological Polar Surface Area / Solvent-Accessible Surface Area'] = df['ROMol'].apply(calculate_tpsa_ratio)
df['Fraction Aromatic Rings'] = df['ROMol'].apply(Chem.rdMolDescriptors.CalcNumAromaticRings) / df['ROMol'].apply(lambda mol: mol.GetNumBonds())
df['Fraction Rotatable Bonds'] = df['ROMol'].apply(Chem.rdMolDescriptors.CalcNumRotatableBonds) / df['ROMol'].apply(lambda mol: mol.GetNumBonds())
df['Fraction Hydrogen Bond Acceptors'] = df['ROMol'].apply(Chem.rdMolDescriptors.CalcNumLipinskiHBA) / df['ROMol'].apply(lambda mol: mol.GetNumBonds())
df['Fraction Hydrogen Bond Donors'] = df['ROMol'].apply(Chem.rdMolDescriptors.CalcNumLipinskiHBD) / df['ROMol'].apply(lambda mol: mol.GetNumBonds())
df['Fraction sp3'] = df['ROMol'].apply(lambda mol: rdMolDescriptors.CalcFractionCSP3(mol))


df['RULE OF FIVE'] = df.apply(rule_of_five, axis=1)

# Replace NaN and INF with a specific value, e.g., empty string ''
df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace INF with NaN
df.fillna('', inplace=True)  # Replace NaN with an empty string

if args.output_excel:
    try:
        PandasTools.SaveXlsxFromFrame(df, f"./{args.input_file.split('.')[0]}_analysis.xls", molCol='ROMol', size=(150,150))
        print("XLS file saved successfully.")
    except:
        print("Error saving the XLS file.")

# Save the updated DataFrame to a new CSV file
df.drop(columns=['ROMol'], inplace=True)
output_file = f"{args.input_file.split('.')[0]}_analysis.txt"
df.to_csv(output_file, sep='\t', index=False)

print("Calculations completed and saved to:", output_file)
