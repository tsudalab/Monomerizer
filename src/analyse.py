#!/usr/bin/env python3

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, DataStructs, Descriptors
import os, sys, requests, tqdm, re, argparse
from collections import defaultdict
import xml.etree.ElementTree as ET

def add_canonical_smiles(df):
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
        "CC(C)[C@@H](C(=O)O)N",                    # Valine (V)
        "CC(C)C[C@@H](C(=O)O)N",                   # Leucine (L)
        "CC[C@H](C)[C@@H](C(=O)O)N",               # Isoleucine (I)
        "C[C@H]([C@@H](C(=O)O)N)O",                # Threonine (T)
        "C([C@@H](C(=O)O)N)S",                     # Cysteine (C)
        "C([C@@H](C(=O)O)N)O",                     # Serine (S)
        "C[C@@H](C(=O)O)N",                        # Alanine (A)
        "C(C(=O)O)N"                               # Glycine (G)
    ]
    one_letter_codes = ['W','R','H','P','K','M','N','Q','E','D','Y','F','V','L','I','T','C','S','A','G']

    canonical_df = pd.DataFrame({
        'ID': one_letter_codes,
        'SMILES': canonical_smiles_list,
        'CANONICAL': ['True'] * len(canonical_smiles_list),
        'TERMINAL': ['NotTer'] * len(canonical_smiles_list),
        'ROMol': [Chem.MolFromSmiles(smi) for smi in canonical_smiles_list]
    })

    return pd.concat([df, canonical_df], ignore_index=True)

def cal_tanimoto(mol):
    l_glycine = Chem.MolFromSmiles("C(C(=O)O)N")
    fp1 = rdMolDescriptors.GetMorganFingerprint(mol, 2)
    fp2 = rdMolDescriptors.GetMorganFingerprint(l_glycine, 2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def fetch_pubchem_name(smiles):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/property/Title/JSON"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data['PropertyTable']['Properties'][0].get('Title', 'NULL')
    except (requests.exceptions.RequestException, KeyError, IndexError):
        return "NULL"

def fetch_chembl_similarity(smiles, similarity_threshold=100):
    try:
        url = f"https://www.ebi.ac.uk/chembl/api/data/similarity/{smiles}/{similarity_threshold}"
        response = requests.get(url)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        chembl_ids = [m.find('.//molecule_chembl_id').text for m in root.findall('.//molecule') if m.find('.//molecule_chembl_id') is not None]
        return chembl_ids if chembl_ids else ["NULL"]
    except requests.exceptions.RequestException:
        return ["NULL"]

def fetch_names(smiles):
    pubchem_name = fetch_pubchem_name(smiles)
    chembl_names = fetch_chembl_similarity(smiles)
    return pubchem_name, ",".join(chembl_names)

def fetch_rdkit_properties(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ["NULL"] * 7
        weight = Descriptors.ExactMolWt(mol)
        clogp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        charge = Chem.GetFormalCharge(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        h_donors = Descriptors.NumHDonors(mol)
        h_acceptors = Descriptors.NumHAcceptors(mol)
        return [weight, clogp, tpsa, charge, rotatable_bonds, h_donors, h_acceptors]
    except Exception:
        return ["NULL"] * 7

def count_monomers(mols_df):
    monomers_dict = defaultdict(int)
    for sequence in mols_df['SEQUENCE']:
        if isinstance(sequence, str) and len(sequence) > 0:
            tokens = re.findall('[A-Z][^A-Z]*', sequence)
            for token in tokens:
                monomers_dict[token] += 1
    return monomers_dict

def main():
    parser = argparse.ArgumentParser(description='Analyse non-natural amino acids (NNAA) from PubChem.')
    parser.add_argument('--input_dir', help='Input directory containing the monomer data.', default='data/tmp')
    parser.add_argument('--mols_file', help='File name relative to input_dir.', default='standard/sequences_standardized.txt')
    parser.add_argument('-fetch_names', help='Fetch names from PubChem and ChEMBL.', action='store_true')
    parser.add_argument('--target_type', help='Type of target: ncAAs or peptides?', default='ncAAs')
    parser.add_argument('--output_file', help='Output CSV file name.', default='analysis.csv')
    args = parser.parse_args()

    mols_path = args.mols_file
    output_path = os.path.join(args.input_dir, args.output_file)

    df = pd.read_csv(mols_path, sep='\t')
    df = df.dropna(subset=['SMILES']).drop_duplicates(subset=['SMILES'])
    df['ROMol'] = df['SMILES'].apply(Chem.MolFromSmiles)

    if args.fetch_names:
        df[['PUBCHEM_NAME', 'CHEMBL_NAMES']] = df['SMILES'].apply(fetch_names).tolist()

    df['Tanimoto_to_Glycine'] = df['ROMol'].apply(cal_tanimoto)
    df[['MolWt', 'LogP', 'TPSA', 'FormalCharge', 'RotatableBonds', 'HydrogenDonors', 'HydrogenAcceptors']] = df['SMILES'].apply(fetch_rdkit_properties).tolist()

    df.to_csv(output_path, index=False)
    print(f"Processing completed. Results saved to {output_path}")

if __name__ == "__main__":
    main()
