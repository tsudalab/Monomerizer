# Monomerizer

Monomerizer (or SMILES2Seq, #SMILES2FASTA) is a pipeline that converts peptides and peptidomimetics, represented as SMILES (chemical formulae), into sequences of amino acids and terminal modifications.

For more information, visit our paper:  [Monomerizer Documentation](https://...).

![alt text](TOC.png)

To use the output data to finetune our foundation language model for peptidomimetics, visit: [GPepT](https://huggingface.co/Playingyoyo/GPepT)

---

## Usage

To run a Monomerizer demo, use the following command:

```
python3 run_pipeline.py --input_file demo/example_smiles.txt
```

- By default, results will be saved to the `output/<datetime>` directory. The raw directory contains the raw result, and the standard directory contains the sequences after standardizing them to the standard dictionary accepted by GPepT.
- Replace `demo/example_smiles.txt` with the path to your input file containing SMILES strings.  (The input file must follow the format of the example files in the `demo` directory.)

---

### Optional arguments

- `--output_dir <path>`
- `--min_amino_acids <int>`: Minimum number of amino acids required for processing. Default is `3`.
- `--batch_size <int>`: Number of SMILES to process in each batch. Default is `100`.
- `--max_workers <int>`: Maximum number of parallel workers. Default is the number of available CPU cores.
- `-draw`: Draws output file like this.

![alt text](demo/example.svg)

---
