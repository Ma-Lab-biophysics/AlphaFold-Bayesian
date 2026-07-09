This example shows how to run AvTraj to estimate distances between two FRET-labeled residues. AvTraj computes accessible volumes for the donor and acceptor dyes and converts the dye-distance results to model FRET efficiencies.

Required external Python packages:
- mdtraj and PyYAML: `conda install -c conda-forge mdtraj pyyaml`
- avtraj v0.0.8: `pip install avtraj`

If the active environment uses Python 3.x, AvTraj v0.0.8 requires small Python 3 compatibility patches (AvTraj v0.0.8 was originally written for Python 2.x):
`python patch_avtraj_py3.py`

Input data for AvTraj:
- `labeling.fps.json`: AvTraj input file specifying the FRET probe properties
- `Structures/frame_*.pdb`: PDB frame files for the structural ensemble
- `Structures/frame_0.pdb`: topology file used to load the frame series

Run the AvTraj calculations:
```bash
cd examples/mean_var/Running-Avtraj
python run_avtraj.py
```

Output data:
- `avtraj_rDAE_values.dat`: one rDAE distance value per input structure
- `fret.input`: corresponding FRET values that can be used for bayes-infer

In this example, the parameters correspond to Alexa Fluor 555 and Alexa Fluor 647, each with a linker length of 21 Å. R0 = 51.0 Å. The probes are attached to residues 100 and 473 of UvrD.
