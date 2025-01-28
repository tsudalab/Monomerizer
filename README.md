# Monomerizer

Monomerizer converts peptides and peptidomimetics, represented as SMILES (chemical formulae), into sequences of amino acids and terminal modifications.

For more information, visit:  
[Monomerizer Documentation](https://...)

#SMILES2Seq #SMILES2FASTA #Non-canonical-aminoacids
---

## Usage

To run a Monomerizer demo, use the following command:

```
python3 run_pipeline.py --input_file demo/example_smiles.txt
```

- By default, results will be saved to the `output/<datetime>` directory.  
- Replace `demo/example_smiles.txt` with the path to your input file containing SMILES strings.  

### Input File Requirements
The input file must follow the format of the example files in the `demo` directory. Ensure your file adheres to these standards for successful processing.  

---

### Optional arguments

- `-process_cyclic`: Include this flag to process cyclic peptides.
- `--min_amino_acids <int>`: Minimum number of amino acids required for processing. Default is `3`.
- `--batch_size <int>`: Number of SMILES to process in each batch. Default is `100`.
- `--output_dir <path>`: Directory to save the output results.
- `--max_workers <int>`: Maximum number of parallel workers. Default is the number of available CPU cores.

---
