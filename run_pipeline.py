import argparse
import os
import subprocess
import sys
import datetime

def run_pipeline(input_file, output_dir, monomerizer_args=None):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Run monomerizer.py with its arguments
    print(f"Running monomerizer.py... Input: {input_file}, Output: {output_dir}")
    monomerizer_command = [sys.executable, "src/monomerizer.py", "--input_file", input_file, "--output_dir", output_dir]
    if monomerizer_args:
        monomerizer_command.extend(monomerizer_args)
    subprocess.run(monomerizer_command, check=True)

    # Step 2: Run standardizer.py with its arguments
    print("Running standardizer.py...")
    standardizer_command = [sys.executable, "src/standardizer.py", "--output_dir", output_dir]
    subprocess.run(standardizer_command, check=True)

    # Step 3: Run prepare_GPepT_data.py to process sequences
    print("Running prepare_GPepT_data.py...")
    prepare_gpept_data_command = [sys.executable, "src/prepare_GPepT_data.py", "--output_dir", output_dir]
    subprocess.run(prepare_gpept_data_command, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a pipeline of programs sequentially.")
    
    # Add arguments
    parser.add_argument("--input_file", default="demo/example_smiles.txt", help="Input file for the pipeline")
    parser.add_argument("--output_dir", default=f"output/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}", help="Output directory")
    parser.add_argument("--process_cyclic", action="store_true", help="Process cyclic compounds")
    parser.add_argument("--min_amino_acids", type=int, help="Minimum number of amino acids required")
    parser.add_argument("--batch_size", type=int, help="Batch size for processing")
    parser.add_argument("--max_workers", type=int, help="Maximum number of workers for parallel processing")
    parser.add_argument("-draw", action="store_true", help="Draw the molecules")

    args = parser.parse_args()

    # Prepare extra arguments for monomerizer.py
    monomerizer_args = []
    if args.process_cyclic:
        monomerizer_args.append("-process_cyclic")
    if args.min_amino_acids:
        monomerizer_args.extend(["--min_amino_acids", int(args.min_amino_acids)])
    if args.batch_size:
        monomerizer_args.extend(["--batch_size", str(args.batch_size)])
    if args.max_workers:
        monomerizer_args.extend(["--max_workers", str(args.max_workers)])
    if args.draw:
        monomerizer_args.append("-draw")

    # Run the pipeline
    run_pipeline(args.input_file, args.output_dir, monomerizer_args=monomerizer_args)
