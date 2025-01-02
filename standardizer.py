#!/usr/bin/env python3

import pandas as pd
import re

# Paths for input files
standard_ncAAs_file = 'ncAAs_standard.txt'
raw_ncAAs_file = 'output/ncAAs_raw.txt'
sequence_file = 'output/sequences_raw.txt'
id_mapping_output = 'output/ncAAs_raw2standard.txt'
relabeled_ncAAs_output = 'output/ncAAs_standardized.txt'
relabeled_sequence_output = 'output/sequences_standardized.txt'

# Load the analysis DataFrame
standard_ncAAs = pd.read_csv(standard_ncAAs_file, sep='\t')
raw_ncAAs = pd.read_csv(raw_ncAAs_file, sep='\t')

# Remove rows whose 'ID' does not start with 'X'
raw_ncAAs = raw_ncAAs[raw_ncAAs['ID'].str.startswith('X')]

# Dictionary to store old and new IDs
id_map = {}

# Function to relabel IDs of the current_ncAAs DataFrame according to the standard_ncAAs DataFrame ID with the same SMILES
def relabel_id(row):
    old_id = row['ID']
    # Find the row in standard_ncAAs with the same SMILES
    match = standard_ncAAs[standard_ncAAs['SMILES'] == row['SMILES']]
    if not match.empty:
        new_id = match['ID'].values[0]
        id_map[old_id] = new_id  # Record old and new ID mapping
        return new_id
    else:
        return "[UNK]"

# Apply the function to relabel IDs and store old-new ID mappings
raw_ncAAs['ID'] = raw_ncAAs.apply(relabel_id, axis=1)

# Save the ID mapping
id_map_df = pd.DataFrame(list(id_map.items()), columns=['raw_ID', 'standard_ID'])
id_map_df.to_csv(id_mapping_output, sep='\t', index=False)

raw_ncAAs.to_csv(relabeled_ncAAs_output, sep='\t', index=False)

# Load the sequence file
sequence_df = pd.read_csv(sequence_file, sep='\t')

# Drop rows whose 'SEQUENCE' is NaN
sequence_df = sequence_df.dropna(subset=['SEQUENCE'])

# Function to apply the relabeling in the SEQUENCE column
def relabel_sequence(sequence):
    # Split the sequence by capital letters, which separates each ID
    tokens = re.split(r"(?=[A-Z])", sequence)
    # Replace each token if it matches an old ID in the map
    relabeled_tokens = [id_map.get(token, token) for token in tokens]
    # If '[NA]' is in the relabeled tokens, return an empty string
    if '[NA]' in relabeled_tokens:
        return ''
    # Reassemble the sequence
    return ''.join(relabeled_tokens)

# Apply relabeling to each sequence
sequence_df['SEQUENCE'] = sequence_df['SEQUENCE'].apply(relabel_sequence)

# Save the relabeled sequences
sequence_df.to_csv(relabeled_sequence_output, sep='\t', index=False)

print("Relabeling complete.")
print(f"ID mapping saved to: {id_mapping_output}")
print(f"Relabeled sequences saved to: {relabeled_sequence_output}")
