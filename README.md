# Vx-passivation-mlff

X-type ligand passivation at defect sites on nanocrystal surfaces with M3GNet-based structural relaxation and adsorption-energy evaluation.

## Repository

Source code and updates are available at:

`https://github.com/kushalsamanta/Vx-passivation-mlff`

## Overview

This application is built for automated ligand passivation studies on defective nanocrystal surfaces. It enables users to:

- upload a defective nanocrystal structure,
- upload one ligand or a batch of ligands,
- identify compatible X-type ligand head groups such as phosphonic, sulfonic, and carboxylic motifs,
- generate multiple candidate attached NC+ligand configurations,
- optionally relax the isolated ligand and the NC+ligand structures using an uploaded MatGL/M3GNet model,
- evaluate the adsorption energy for the most stable relaxed configuration.

## Key capabilities

- Single-ligand and batch-ligand workflows
- Support for `.vasp`, `.cif`, `.poscar`, and `.contcar` files
- Automatic generation of multiple ligand orientations near the defect site
- Optional M3GNet relaxation for:
  - the neutral isolated ligand,
  - each selected NC+ligand configuration
- Automatic selection of the lowest-energy relaxed NC+ligand structure
- Downloadable result files for summary and structures

## Required inputs

### 1. Defective nanocrystal structure

Upload a defective nanocrystal structure in one of the following formats:

- `.vasp`
- `.cif`
- `.poscar`
- `.contcar`

### 2. Ligand input

Users may provide either:

- one ligand structure file,
- multiple ligand structure files, or
- one ZIP archive containing multiple ligand files

Supported ligand formats:

- `.vasp`
- `.cif`
- `.poscar`
- `.contcar`

### 3. Optional MatGL/M3GNet model ZIP

If structural relaxation is required, upload one ZIP file containing a trained MatGL/M3GNet model.

The ZIP file should contain:

- `model.pt`
- `state.pt`
- `model.json`

Accepted layouts include:

```text
model.zip
├─ model.pt
├─ state.pt
└─ model.json
```

or

```text
model.zip
└─ out/
   ├─ model.pt
   ├─ state.pt
   └─ model.json
```

## Installation

Clone the repository:

```bash
git clone https://github.com/kushalsamanta/Vx-passivation-mlff.git
cd Vx-passivation-mlff
```

Install the required Python packages:

```bash
pip install --no-cache-dir -r requirements.txt
```

## Running the application locally

Launch the Gradio application with:

```bash
python app.py
```

After startup, Gradio prints a local URL in the terminal. Open that link in a browser to use the app.

## Running the application in Google Colab

This project can also be used from Google Colab.

### Step 1: Open a new Colab notebook

Create a fresh notebook in Google Colab.

### Step 2: Clone the repository

Run:

```python
!git clone https://github.com/kushalsamanta/Vx-passivation-mlff.git
%cd Vx-passivation-mlff
```

### Step 3: Install dependencies

Run:

```python
!pip install --no-cache-dir -r requirements.txt
```

### Step 4: Launch the app

Run:

```python
!python app.py
```

Once the application starts, Colab will display a Gradio link. Open that link to use the interface.

## Standard workflow

A typical calculation proceeds as follows:

1. Upload the defective nanocrystal structure.
2. Upload either one ligand file, multiple ligand files, or one ZIP archive containing ligand files.
3. Optionally upload a trained MatGL/M3GNet model ZIP.
4. Enter the defect-site coordinates and relevant workflow parameters.
5. Run the attachment and relaxation workflow.
6. Review logs, energies, and downloadable output files.

## Output files

The application can generate the following outputs.

### Summary CSV

The summary table may include:

- ligand name
- detected ligand family
- number of attached configurations
- best relaxed configuration name
- best relaxed NC+ligand energy
- relaxed neutral ligand energy
- adsorption energy
- workflow status

### ZIP of attached NC+ligand configurations

Contains candidate attached structures before relaxation.

### ZIP of best relaxed NC+ligand structures

Contains the lowest-energy relaxed structure for each ligand, when relaxation succeeds.

## Adsorption energy

The adsorption energy is evaluated as:

```text
E_ads = E[NC+VCl+ligand] - (E[NC+VCl] + E[ligand] - 1/2 E[H2])
```

with:

```text
1/2 E[H2] = -3.393237 eV
```

## Notes for users

- The application supports both single-ligand and batch-ligand calculations.
- When relaxation is enabled and a valid model ZIP is supplied, the app relaxes the separate neutral ligand.
- The app also relaxes the selected NC+ligand configurations and retains the most stable relaxed structure.
- If no model ZIP is supplied, the structure-generation workflow can still be used without relaxation.

## Troubleshooting

### Model loading fails

Check that:

- the ZIP contains `model.pt`, `state.pt`, and `model.json`,
- these files are either at the ZIP root or inside one folder,
- the installed package versions are compatible with the uploaded model.

### No relaxed structure is produced

Check that:

- relaxation is enabled,
- a valid model ZIP has been uploaded,
- the log window does not report a backend or model-loading error.

### No valid attached configuration is found

Check that:

- the defect-site coordinates are correct,
- the ligand contains a supported passivating head group,
- the uploaded geometry is chemically reasonable.

## Authors

Kushal Samanta<sup>a,b</sup>, Jyoti Bharti<sup>c</sup>, Arun Mannodi-Kanakkithodi<sup>b*</sup>, Dibyajyoti Ghosh<sup>a,c*</sup>

<sup>a</sup> Department of Materials Science and Engineering, Indian Institute of Technology, Delhi-110016, India  
<sup>b</sup> School of Materials Engineering, Purdue University, West Lafayette, IN 47907, United States of America  
<sup>c</sup> Department of Chemistry, Indian Institute of Technology, Delhi-110016, India
