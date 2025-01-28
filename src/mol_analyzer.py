#!/usr/bin/env python3

import pandas as pd
import re
import argparse
import os
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyse sequences.")
    parser.add_argument("--standard_ncAAs_file", required=True, help="Path to the standard ncAAs file.")
    parser.add_argument("--raw_ncAAs_file", required=True, help="Path to the raw ncAAs file.")
    parser.add_argument("--sequence_file", required=True, help="Path to the raw sequence file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save output files.")
    return parser.parse_args()

def check_file_exists(file_path, file_description):
    if not os.path.exists(file_path):
        sys.exit(f"Error: {file_description} '{file_path}' does not exist.")

def main():
    args = parse_arguments()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Paths for output files
    id_mapping_output = os.path.join(args.output_dir, 'ncAAs_raw2standard.txt')
    relabeled_ncAAs_output = os.path.join(args.output_dir, 'ncAAs_standardized.txt')
    relabeled_sequence_output = os.path.join(args.output_dir, 'sequences_standardized.txt')

    # Check if input files exist
    check_file_exists(args.standard_ncAAs_file, "Standard ncAAs file")
    check_file_exists(args.raw_ncAAs_file, "Raw ncAAs file")
    check_file_exists(args.sequence_file, "Sequence file")

    # Load the DataFrames
    standard_ncAAs = pd.read_csv(args.standard_ncAAs_file, sep='\t')
    raw_ncAAs = pd.read_csv(args.raw_ncAAs_file, sep='\t')
    sequence_df = pd.read_csv(args.sequence_file, sep='\t')

    # Check required columns in input files
    required_columns_ncAAs = ['ID', 'SMILES']
    for col in required_columns_ncAAs:
        if col not in standard_ncAAs.columns or col not in raw_ncAAs.columns:
            sys.exit(f"Error: Missing required column '{col}' in ncAAs files.")

    if 'SEQUENCE' not in sequence_df.columns:
        sys.exit("Error: Missing 'SEQUENCE' column in sequence file.")

    # Filter rows whose 'ID' starts with 'X'
    raw_ncAAs = raw_ncAAs[raw_ncAAs['ID'].str.startswith('X')]

    # Dictionary to store old and new IDs
    id_map = {}

    # Function to relabel IDs based on standard_ncAAs
    def relabel_id(row):
        old_id = row['ID']
        match = standard_ncAAs[standard_ncAAs['SMILES'] == row['SMILES']]
        if not match.empty:
            new_id = match['ID'].values[0]
            id_map[old_id] = new_id
            return new_id
        else:
            return "[UNK]"

    # Relabel IDs and create ID mapping
    raw_ncAAs['ID'] = raw_ncAAs.apply(relabel_id, axis=1)

    # Save the ID mapping
    id_map_df = pd.DataFrame(list(id_map.items()), columns=['raw_ID', 'standard_ID'])
    id_map_df.to_csv(id_mapping_output, sep='\t', index=False)

    # Save relabeled ncAAs
    raw_ncAAs.to_csv(relabeled_ncAAs_output, sep='\t', index=False)

    # Drop rows with NaN in 'SEQUENCE'
    sequence_df = sequence_df.dropna(subset=['SEQUENCE'])

    # Function to relabel sequences
    def relabel_sequence(sequence):
        tokens = re.split(r"(?=[A-Z])", sequence)
        relabeled_tokens = [id_map.get(token, token) for token in tokens]
        if '[UNK]' in relabeled_tokens:
            return ''
        return ''.join(relabeled_tokens)

    # Apply relabeling to sequences
    sequence_df['SEQUENCE'] = sequence_df['SEQUENCE'].apply(relabel_sequence)

    # Remove rows with empty sequences
    sequence_df = sequence_df[sequence_df['SEQUENCE'] != '']

    # Save relabeled sequences
    sequence_df.to_csv(relabeled_sequence_output, sep='\t', index=False)

    print("Relabeling complete.")
    print(f"ID mapping saved to: {id_mapping_output}")
    print(f"Relabeled ncAAs saved to: {relabeled_ncAAs_output}")
    print(f"Relabeled sequences saved to: {relabeled_sequence_output}")

if __name__ == "__main__":
    main()
