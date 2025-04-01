import random
import argparse
import pandas as pd
import os

# Set up argument parser
parser = argparse.ArgumentParser(description="Process sequences from an input file and split them into two output files.")
parser.add_argument('--output_dir', type=str, default='output/tmp', help="Directory containing the input file")
args = parser.parse_args()

# Define input file and output file paths
input_file = os.path.join(args.output_dir, 'standard/sequences_standardized.txt')
os.makedirs(os.path.join(args.output_dir, 'for_GPepT'), exist_ok=True)
output_file_90 = os.path.join(args.output_dir, 'for_GPepT/train90.txt')
output_file_10 = os.path.join(args.output_dir, 'for_GPepT/val10.txt')

# Check if the input file exists
if not os.path.exists(input_file):
    # No ncAAs?
    input_file = os.path.join(args.output_dir, 'raw/sequences_raw.txt')
    if not os.path.exists(input_file):
        print(f"Error: The input file '{input_file}' does not exist.")
        exit(1)

# Read the input file into a pandas DataFrame
df = pd.read_csv(input_file, sep='\t')

# Extract sequences and add <endoftext> to each
sequences = df['SEQUENCE'].apply(lambda x: x + '<|endoftext|>')

# Shuffle the sequences to randomize the split
sequences = sequences.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the sequences into 90% and 10%
split_index = int(0.9 * len(sequences))
sequences_90 = sequences[:split_index]
sequences_10 = sequences[split_index:]

# Write the sequences to the output files
sequences_90.to_csv(output_file_90, index=False, header=False)
sequences_10.to_csv(output_file_10, index=False, header=False)

print(f"Data has been successfully split into {output_file_90} and {output_file_10}")
