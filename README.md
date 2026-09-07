# Bayesian Inference of Protein Conformational Ensembles by Integrating AlphaFold-based Modeling with Sparse Experimental Data

This tool provides an AF sampling module and two Bayesian inference modules:

1. **AF-sample**: script generation for AlphaFold sampling with varying MSA depth and random seeds.

2. **mean-var**: mean/variance fitting for one-dimensional observables where the experimental restraints are a mean and a variance, for example FRET `<E>` and `Var(E)`. It fits the nonlinear variance expression directly.

3. **multi-d**: multi-dimensional restraint workflow. The example prepares input files from NMR-PRE data and optimizes the model weights.

## Layout

```text
AlphaFold-Bayesian/
├── pyproject.toml
├── environment.yml
├── README.md
├── THIRD_PARTY_NOTICES.md
├── MODEL_CARD.md
├── src/af_sample/
│   ├── __init__.py
│   └── cli.py
├── src/bayes_infer/
│   ├── cli.py
│   ├── mean_var/
│   │   └── mean_variance.py
│   └── multi_d/
│       ├── generic_io.py
│       ├── generic_protocol.py
│       ├── nmr_prep.py
│       └── backends/
│           ├── bioen_common.py
│           └── bioen_log_weights.py
├── examples/
│   └── mean_var/
└── tests/
```

## Install

```bash
conda env create -f environment.yml
conda activate af-bayes
```

## AlphaFold sampling

`AF-sample` writes a ColabFold run script for AlphaFold sampling by varying MSA depth and the number of random seeds. It generates `run_AF.sh` in the current directory; the user runs the script manually.

A cluster or server with multiple NVIDIA GPUs is recommended to run ColabFold. Install ColabFold locally before running the generated script (CUDA 12.1 or later is required; using the Pixi package to install LocalColabFold is recommended). See installation details [here](https://github.com/YoshitakaMo/localcolabfold/).

Generate a ColabFold run script:

```bash
AF-sample make-script \
  --inputfile input.fasta \
  --outputdir AF_output
```

By default this uses `--max-seq 512`, `--max-extra-seq 1024`, `--num-recycle 3`, `--random-seed 0`, `--num-seeds 100`, `--num-relax 500`, and `--relax-max-iterations 100`. If `--nrelax` is not set, it defaults to five times `--nseeds`.

Run the generated script:

```bash
bash run_AF.sh
```

After `colabfold_batch` finishes, `run_AF.sh` collects relaxed PDB files into:

```text
./Structures/
```

Customize the sampling settings:

```bash
AF-sample make-script \
  --inputfile input.fasta \
  --outputdir AF_output \
  --max-seq 256 \
  --max-extra-seq 512 \
  --ncycle 3 \
  --seed 0 \
  --nseeds 100 \
  --nrelax 500 \
  --rsteps 100
```

## Pre-processing for Bayesian inference

For 1D smFRET (mean-var), use AvTraj to calculate FRET efficiencies for the AlphaFold models. More details are available in `./examples/mean_var/Running-Avtraj`.

Input format for mean-var:

```text
theta
mean  mean_SE
variance  variance_SE
prior_weight_i  observable_i
...
```

For NMR-PRE data (multi-d), use the following script to prepare the multi-dimensional distance restraints for the subsequent Bayesian inference.

```bash
bayes-infer multi-d make-data \
  --raw-xls wt_tau_values_MTSL.xls \
  --r1rho-xls R1rho_Values.xls \
  --structures-dir Structures \
  --out-dir output_multi_d
```

By default, this writes the generic multi-d input files to `output_multi_d/multi_d_input`. Existing `BioEN_Files` directories from older runs remain readable when passed with `--data-dir`.

## mean-var module

Run for one theta value:

```bash
bayes-infer mean-var run open.input \
  -o output_open.dat \
  --theta-log theta_open.log
```

Run a theta scan:

```bash
bayes-infer mean-var theta-scan \
  --templates open.input \
  --start 0.1 \
  --stop 10 \
  --num-theta 200 \
  --spacing linear \
  --output-root mean_var_theta_scan
```

## multi-d module

Run for one theta value:

```bash
bayes-infer multi-d run \
  --data-dir output_multi_d/multi_d_input \
  --theta 3000 \
  --out-dir output_multi_d/theta_3000
```

Run a theta scan:

```bash
bayes-infer multi-d theta-scan \
  --data-dir output_multi_d/multi_d_input \
  --start 1e-2 \
  --stop 1e5 \
  --num-theta 10 \
  --spacing log \
  --output-root output_multi_d/theta_scan
```

<!-- By default, `multi-d theta-scan` creates each theta folder and runs the local log-weight optimizer. Add `--no-run` to only create the folders and command files. -->

A model card (https://huggingface.co/docs/hub/model-cards) is provided in MODEL_CARD.md.

## Third-party components

BioEn (https://github.com/bio-phys/BioEn/) is closely related to our method, and we adapt selected BioEn components as documented in THIRD_PARTY_NOTICES.md.

## Licensing

This package is distributed under the GNU General Public License v3.0 or later. See `THIRD_PARTY_NOTICES.md` for BioEn-adapted components.
