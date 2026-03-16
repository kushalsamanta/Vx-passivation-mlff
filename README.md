
# Vx-passivation-mlff

X-type ligand passivation at defect sites on nanocrystal surfaces with M3GNet-based structural relaxation and adsorption energy evaluation.

## What this repository contains

This repository provides a Gradio-based application for:

- attaching X-type ligands to defective nanocrystal surfaces,
- generating multiple candidate NC+ligand geometries,
- optionally relaxing the neutral ligand and attached NC+ligand structures with a trained M3GNet / MatGL model,
- estimating adsorption energies, and
- exporting summary tables and downloadable structure files.

Repository files:

- `app.py` — the main Gradio application
- `requirements.txt` — Python dependencies for Google Colab or local installation
- `README.md` — setup and usage instructions

---

## What to do next after uploading these files to GitHub

Since you have already uploaded:

- `app.py`
- `requirements.txt`
- `README.md`

these are the next steps.

### Option 1: Use it through Google Colab

This is the easiest route if you want to share the app with others and let them run it on their own Colab session.

#### Step 1
Open your GitHub repository and copy its URL.

#### Step 2
Create a new Google Colab notebook.

#### Step 3
In the first cell, clone your repository:

```python
!git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
%cd YOUR-REPO-NAME
```

Replace:

- `YOUR-USERNAME` with your GitHub username
- `YOUR-REPO-NAME` with your repository name

#### Step 4
Install the required packages from `requirements.txt`:

```python
!pip install --no-cache-dir -r requirements.txt
```

If needed, restart the runtime once after installation.

#### Step 5
Run the app:

```python
!python app.py
```

If your app uses Gradio public sharing, it will print a temporary public link in the notebook output.

#### Step 6
Use the app interface in the generated link.

Users can then:

- upload the defective nanocrystal structure,
- upload one ligand or multiple ligands,
- optionally upload the trained M3GNet model ZIP,
- run the workflow,
- download the generated structures and summary files.

---

### Option 2: Run locally on your computer

If someone wants to run the code locally:

#### Step 1
Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME
```

#### Step 2
Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

#### Step 3
Install dependencies:

```bash
pip install --no-cache-dir -r requirements.txt
```

#### Step 4
Run the app:

```bash
python app.py
```

Then open the local Gradio URL shown in the terminal.

---

## Recommended Google Colab workflow for users

For users who only want to run calculations and do not want to manually set up anything complicated, follow these steps.

### Step 1: Open Colab
Go to Google Colab and create a new notebook.

### Step 2: Clone the repository
Paste and run:

```python
!git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
%cd YOUR-REPO-NAME
```

### Step 3: Install packages
Paste and run:

```python
!pip install --no-cache-dir -r requirements.txt
```

### Step 4: Launch the app
Paste and run:

```python
!python app.py
```

### Step 5: Use the interface
After the Gradio link appears:

- open the link,
- upload structures,
- run the workflow,
- download the output files.

---

## Input files expected by the app

### 1. Defective nanocrystal structure
Accepted formats include:

- POSCAR / VASP
- CIF
- POSCAR-like files

### 2. Ligand input
You may provide:

- one single ligand file, or
- multiple ligand files, or
- one ZIP archive containing many ligand files.

### 3. Optional trained M3GNet / MatGL model ZIP
The uploaded ZIP should contain:

- `model.pt`
- `state.pt`
- `model.json`

Supported ZIP layouts:

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

---

## What the app does

For each ligand, the workflow can:

1. identify the ligand head-group family,
2. generate possible attached NC+ligand structures,
3. save multiple candidate configurations,
4. relax the neutral ligand with M3GNet,
5. relax the attached NC+ligand structures with M3GNet,
6. select the lowest-energy relaxed NC+ligand configuration,
7. compute adsorption energy,
8. export a summary CSV and downloadable ZIP files.

---

## Output files

The application generates:

- `summary.csv` — summary of all ligands and calculated energies
- ZIP of all attached NC+ligand candidate structures
- ZIP of best relaxed NC+ligand structures

Depending on the workflow settings, the outputs may include:

- best relaxed NC+ligand energy,
- relaxed neutral ligand energy,
- adsorption energy,
- selected best configuration index.

---

## Adsorption energy expression

The adsorption energy is evaluated as:

```text
E_ads = E[NC+VCl+ligand] - (E[NC+VCl] + E[ligand] - 1/2 E[H2])
```

with:

```text
1/2 E[H2] = -3.393237 eV
```

---

## Example Colab notebook cells

### Cell 1: Clone repository

```python
!git clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
%cd YOUR-REPO-NAME
```

### Cell 2: Install dependencies

```python
!pip install --no-cache-dir -r requirements.txt
```

### Cell 3: Run the app

```python
!python app.py
```

---

## If installation fails in Colab

Try these steps:

1. Restart the runtime.
2. Run the installation cell again.
3. Ensure that `requirements.txt` is present in the repository root.
4. Make sure the uploaded M3GNet ZIP contains the correct files.

---

## Suggested repository structure

```text
Vx-passivation-mlff/
├── app.py
├── requirements.txt
└── README.md
```

---

## Notes for sharing with others

If you want others to use your workflow easily:

- keep `app.py`, `requirements.txt`, and `README.md` in the repository root,
- share the GitHub repository link,
- also share one prepared Google Colab notebook that clones the repository and runs it.

This is usually the easiest method for collaborators.

---

## Citation / acknowledgement

If you use this workflow in academic work, please cite the corresponding manuscript, software model, and dataset as appropriate.

---

## Contact

For questions, issues, or collaboration, please contact the repository owner.
