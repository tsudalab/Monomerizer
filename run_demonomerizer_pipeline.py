import argparse
import os
import subprocess
import sys
import datetime

def run_pipeline(sequence_file, output_dir, demonomerized_file, demonomerizer_args=None, analyse_args=None):
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Run demonomerizer.py
    print(f"Running demonomerizer.py... Input: {sequence_file}")
    demonomerizer_command = [
        sys.executable, "src/demonomerizer.py",
        "--sequence_file", sequence_file,
        "--NNAA_file", "dictionary.txt",
        "--batch_size", "8",
        "--output_dir", output_dir,
        "--demonomerized_file", demonomerized_file
    ]

    subprocess.run(demonomerizer_command, check=True)

    demonomerized_path = os.path.join(output_dir, demonomerized_file)

    # Step 2: Run analyse.py
    print("Running analyse.py...")
    analyse_command = [
        sys.executable, "src/analyse.py",
        "--mols_file", demonomerized_path,
        "--input_dir", output_dir,
        "--target_type", "peptides",
    ]
    if analyse_args:
        analyse_command.extend(analyse_args)

    subprocess.run(analyse_command, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the demonomerizer pipeline.")
    parser.add_argument("--sequence_file", default="demonomerized.txt", help="Input sequence file")
    parser.add_argument("--output_dir", default=f"output/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}", help="Directory to store output")
    parser.add_argument("--demonomerized_file", default="sequences_standardized.txt", help="Output demonomerized file name")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for demonomerizer.py")
    parser.add_argument("-fetch_names", action="store_true", help="Fetch names from PubChem in analyse.py")
    parser.add_argument("--target_type", default="ncAAs", help="Target type: ncAAs or peptides")

    args = parser.parse_args()

    # Args for demonomerizer
    demonomerizer_args = ["--NNAA_file", "dictionary.txt", "--batch_size", str(args.batch_size)]

    # Args for analyse
    analyse_args = []
    if args.fetch_names:
        analyse_args.append("-fetch_names")
    if args.target_type:
        analyse_args.extend(["--target_type", args.target_type])

    run_pipeline(args.sequence_file, args.output_dir, args.demonomerized_file, demonomerizer_args, analyse_args)
