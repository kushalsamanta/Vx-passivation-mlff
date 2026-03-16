# Vx-passivation-mlff

X-type ligand passivation at defect sites on nanocrystal surfaces with M3GNet-based structural relaxation and adsorption energy evaluation.

## Overview

This repository provides a Gradio-based application for:

- attaching X-type ligands to defective nanocrystal surfaces,
- generating multiple candidate NC+ligand geometries,
- optionally relaxing the neutral ligand and attached NC+ligand structures with a trained M3GNet / MatGL model,
- estimating adsorption energies, and
- exporting summary tables and downloadable structure files.

Repository URL:

```text
https://github.com/kushalsamanta/Vx-passivation-mlff
```

## Repository contents

- `app.py` — main Gradio application
- `requirements.txt` — Python dependencies
- `README.md` — setup and usage instructions

## Recommended usage: Google Colab

Google Colab is the simplest way to run this application without a local installation.

### Step 1 — Open a new Colab notebook

Create a new notebook in Google Colab.

### Step 2 — Clone the repository

Run:

```python
!git clone https://github.com/kushalsamanta/Vx-passivation-mlff.git
%cd Vx-passivation-mlff
```

### Step 3 — Install the required packages

Run:

```python
!pip install --no-cache-dir -r requirements.txt
```

If Colab asks for a runtime restart after installation, restart the runtime and then run the notebook cells again from the top.

### Step 4 — Launch the application

Run:

```python
!python app.py
```

When the Gradio server starts, Colab will print a local URL and usually a shareable public Gradio link.

### Step 5 — Use the app

Open the Gradio link and then:

- upload the defective nanocrystal structure,
- upload one ligand file, multiple ligand files, or a ZIP archive containing many ligands,
- optionally upload a trained MatGL / M3GNet model ZIP,
- set the defect-site coordinates and workflow parameters,
- run the workflow,
- download the generated structures and summary files.

## Local installation

For users who prefer to run the application on their own machine:

### Step 1 — Clone the repository

```bash
git clone https://github.com/kushalsamanta/Vx-passivation-mlff.git
cd Vx-passivation-mlff
```

### Step 2 — Create a virtual environment

On Linux or macOS:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bat
python -m venv venv
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install --no-cache-dir -r requirements.txt
```

### Step 4 — Run the app

```bash
python app.py
```

Then open the Gradio URL shown in the terminal.

## Inputs expected by the application

### 1. Defective nanocrystal structure

Accepted formats include:

- POSCAR / VASP
- CIF
- POSCAR-like structure files

### 2. Ligand inputs

The application accepts:

- a single ligand file,
- multiple ligand files, or
- a ZIP archive containing multiple ligand structures.

### 3. Optional trained MatGL / M3GNet model ZIP

The uploaded ZIP should contain these files:

- `model.pt`
- `state.pt`
- `model.json`

Accepted ZIP layouts:

```text
model.zip
├── model.pt
├── state.pt
└── model.json
```

or

```text
model.zip
└── out/
    ├── model.pt
    ├── state.pt
    └── model.json
```

## What the workflow does

For each ligand, the application can:

1. identify the ligand head-group family,
2. generate multiple attached NC+ligand structures,
3. save candidate configurations,
4. relax the neutral ligand with M3GNet,
5. relax the attached NC+ligand structures with M3GNet,
6. select the lowest-energy relaxed NC+ligand configuration,
7. estimate adsorption energy,
8. export a summary CSV and downloadable ZIP files.

## Outputs

The application produces:

- a summary CSV containing ligand-wise results,
- a ZIP archive of attached NC+ligand candidate structures,
- a ZIP archive of best relaxed NC+ligand structures when relaxation succeeds.

## Notes

- The M3GNet relaxation step is optional. The structure-generation workflow can still be used without uploading a model ZIP.
- When a compatible model ZIP is provided, the app attempts to relax both the neutral ligand and the NC+ligand structures.
- Adsorption energies are evaluated from the workflow energies printed in the app and written to the summary file.

## Citation / acknowledgement

If this workflow is useful in academic work, please cite the related research outputs from the repository authors where appropriate.
