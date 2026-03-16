import os
import math
import copy
import csv
import zipfile
import tempfile
import traceback
import random
import inspect
from typing import List

import numpy as np
import gradio as gr
from pymatgen.core import Structure, Molecule
from pymatgen.io.vasp import Poscar
from pymatgen.analysis.adsorption import AdsorbateSiteFinder


# ============================================================
# Suppress a known asyncio cleanup warning on HF Spaces / Gradio
# ============================================================
def _patch_asyncio_event_loop_del():
    try:
        import asyncio.base_events as base_events
        original_del = getattr(base_events.BaseEventLoop, "__del__", None)
        if original_del is None:
            return

        def patched_del(self):
            try:
                original_del(self)
            except ValueError as exc:
                if "Invalid file descriptor" not in str(exc):
                    raise

        base_events.BaseEventLoop.__del__ = patched_del
    except Exception:
        pass


_patch_asyncio_event_loop_del()

# ============================================================
# Global defaults
# ============================================================
# Desired Cs—O_other window and ideal value
CS_D_MIN = 3.0
CS_D_MAX = 4.0
CS_D_IDEAL = 3.5

# Angular search resolution
THETA_STEP_DEG = 10
PHI_STEP_DEG = 10

# Single Pb—O target distance used for placement/orientation search
PB_O_TARGET_DIST = 2.75

# General substrate-vs-ligand non-overlap thresholds
GENERAL_H_MIN_DIST = 1.0
GENERAL_OTHER_MIN_DIST = 1.5

# Specific Cs thresholds near the defect for ligand atoms of any species
CS_H_MIN_DIST = 1.7
CS_OTHER_MIN_DIST = 2.3

# Reference-point non-overlap threshold (defect site + specified Cs points)
REFERENCE_POINT_MIN_DIST = 1.5

MAX_CONFIGS_PER_LIGAND = 10
RANDOM_SEARCH_SEED = 20260312
HALF_H2_ENERGY = -3.393237

SUPPORTED_EXTENSIONS = {".vasp", ".cif", ".poscar", ".contcar"}

DEFAULT_PB = [23.78603232, 23.86694225, 24.00374854]
DEFAULT_CS = [
    [26.01408284369999, 20.8220668374, 26.1983444040],
    [21.1763799991, 20.6568466203, 26.1945442380],
    [20.7875781631, 26.3236480970, 25.6864620438],
]
DEFAULT_E_PRISTINE = -639.069
DEFAULT_FMAX = 0.01
DEFAULT_STEPS = 500


# ============================================================
# Generic helpers
# ============================================================
def parse_xyz_triplet(x, y, z) -> np.ndarray:
    return np.array([float(x), float(y), float(z)], dtype=float)



def get_covalent_radius(el: str) -> float:
    cov_r = {
        "H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66,
        "F": 0.57, "P": 1.07, "S": 1.05, "Cl": 1.02,
    }
    return cov_r.get(el, 1.0)



def rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a) + 1e-15)
    b = b / (np.linalg.norm(b) + 1e-15)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if np.linalg.norm(v) < 1e-12:
        if c > 0.999999:
            return np.eye(3)
        axis = np.array([1.0, 0, 0])
        if abs(np.dot(axis, a)) > 0.9:
            axis = np.array([0, 1.0, 0])
        axis = axis - np.dot(axis, a) * a
        axis /= (np.linalg.norm(axis) + 1e-15)
        K = np.array(
            [[0, -axis[2], axis[1]],
             [axis[2], 0, -axis[0]],
             [-axis[1], axis[0], 0]]
        )
        return np.eye(3) + 2 * K @ K
    s = np.linalg.norm(v)
    K = np.array(
        [[0, -v[2], v[1]],
         [v[2], 0, -v[0]],
         [-v[1], v[0], 0]]
    ) / (s + 1e-15)
    return np.eye(3) + K * s + K @ K * (1 - c)



def check_ads_sub_thresholds(
    structure: Structure,
    n_substrate: int,
    threshold_H: float = GENERAL_H_MIN_DIST,
    threshold_other: float = GENERAL_OTHER_MIN_DIST,
):
    overall_min = float("inf")
    for i in range(n_substrate, len(structure.sites)):
        sp = structure.sites[i].species_string
        ai = np.array(structure.sites[i].coords)
        mind = float("inf")
        for j in range(n_substrate):
            d = np.linalg.norm(ai - np.array(structure.sites[j].coords))
            if d < mind:
                mind = d
        overall_min = min(overall_min, mind)
        if sp == "H":
            if mind < threshold_H:
                return False, overall_min, sp, i
        else:
            if mind < threshold_other:
                return False, overall_min, sp, i
    return True, overall_min, None, None



def check_intramolecular_overlaps(structure: Structure, n_substrate: int, factor: float = 0.8):
    overlaps = []
    for i in range(n_substrate, len(structure.sites)):
        ai = np.array(structure.sites[i].coords)
        for j in range(i + 1, len(structure.sites)):
            aj = np.array(structure.sites[j].coords)
            d = np.linalg.norm(ai - aj)
            r1 = get_covalent_radius(structure.sites[i].species_string)
            r2 = get_covalent_radius(structure.sites[j].species_string)
            if d < factor * (r1 + r2):
                overlaps.append((i, j, d))
    return overlaps



def check_substrate_cs_thresholds(structure: Structure, n_substrate: int):
    """Specific non-overlap rule for all substrate Cs atoms vs all ligand atoms.
    H must stay above CS_H_MIN_DIST, all other ligand atoms above CS_OTHER_MIN_DIST.
    """
    overall_min = float("inf")
    worst_margin = float("inf")
    worst_pair = (None, None, None, None, None)
    cs_indices = [
        j for j in range(n_substrate)
        if structure.sites[j].species_string == "Cs"
    ]
    if not cs_indices:
        return True, overall_min, worst_margin, worst_pair

    ok = True
    for i in range(n_substrate, len(structure.sites)):
        lig_sp = structure.sites[i].species_string
        required = CS_H_MIN_DIST if lig_sp == "H" else CS_OTHER_MIN_DIST
        ai = np.array(structure.sites[i].coords)
        for j in cs_indices:
            d = np.linalg.norm(ai - np.array(structure.sites[j].coords))
            overall_min = min(overall_min, d)
            margin = d - required
            if margin < worst_margin:
                worst_margin = margin
                worst_pair = (j, i, structure.sites[j].species_string, lig_sp, required)
            if d < required:
                ok = False
    return ok, overall_min, worst_margin, worst_pair



def min_distance_points_to_any_ligand_atom(structure: Structure, n_substrate: int,
                                           points: List[np.ndarray]):
    md = float("inf")
    best = (None, None)
    for p_idx, point in enumerate(points):
        point = np.array(point, dtype=float)
        for i in range(n_substrate, len(structure.sites)):
            ai = np.array(structure.sites[i].coords)
            d = np.linalg.norm(ai - point)
            if d < md:
                md = d
                best = (p_idx, i)
    return md, best



def per_point_min_distance_to_any_ligand_atom(structure: Structure, n_substrate: int,
                                              points: List[np.ndarray]):
    out = []
    for p_idx, point in enumerate(points):
        point = np.array(point, dtype=float)
        best_d = float("inf")
        best_i = None
        for i in range(n_substrate, len(structure.sites)):
            ai = np.array(structure.sites[i].coords)
            d = np.linalg.norm(ai - point)
            if d < best_d:
                best_d = d
                best_i = i
        out.append((p_idx, best_d, best_i))
    return out



def check_cs_target_thresholds(structure: Structure, n_substrate: int,
                               cs_targets: List[np.ndarray]):
    """Specific non-overlap rule for the 3 specified Cs target points vs any ligand atom.
    H > CS_H_MIN_DIST, all others > CS_OTHER_MIN_DIST.
    """
    overall_min = float("inf")
    worst_margin = float("inf")
    worst_pair = (None, None, None)
    per_target = []

    for cs_idx, cs in enumerate(cs_targets):
        cs = np.array(cs, dtype=float)
        best_d = float("inf")
        best_i = None
        for i in range(n_substrate, len(structure.sites)):
            ai = np.array(structure.sites[i].coords)
            d = np.linalg.norm(ai - cs)
            if d < best_d:
                best_d = d
                best_i = i

            lig_sp = structure.sites[i].species_string
            required = CS_H_MIN_DIST if lig_sp == "H" else CS_OTHER_MIN_DIST
            overall_min = min(overall_min, d)
            margin = d - required
            if margin < worst_margin:
                worst_margin = margin
                worst_pair = (cs_idx, i, required)
        per_target.append((cs_idx, best_d, best_i))

    ok = worst_margin >= 0.0
    return ok, overall_min, worst_margin, worst_pair, per_target



def _nearest_H_to_O(structure: Structure, O_index: int,
                    primary: float = 1.25, fallback: float = 1.65):
    coords = np.array(structure.cart_coords)
    o = coords[O_index]
    Hs = [i for i in range(len(structure.sites)) if structure.sites[i].species_string == "H"]
    best, best_d = None, 1e9
    for h in Hs:
        d = np.linalg.norm(coords[h] - o)
        if d < primary and d < best_d:
            best, best_d = h, d
    if best is not None:
        return best, best_d
    for h in Hs:
        d = np.linalg.norm(coords[h] - o)
        if d < fallback and d < best_d:
            best, best_d = h, d
    return (best, best_d) if best is not None else (None, None)



def remap_after_deletions(removed_indices: List[int], idx_list: List[int]) -> List[int]:
    removed = sorted(set(removed_indices))

    def shift(idx):
        return idx - sum(1 for r in removed if r < idx)

    return [shift(i) for i in idx_list]



def remove_one_H_from_group(mol_struct: Structure, O_group: List[int], logs: List[str]):
    logs.append(f"    O-group: {O_group}")
    oh_info = []
    for oi in O_group:
        hi, hd = _nearest_H_to_O(mol_struct, oi)
        oh_info.append((oi, hi, hd))
    for oi, hi, hd in oh_info:
        if hi is not None:
            logs.append(f"      O {oi} has H {hi} at {hd:.3f} Å")
        else:
            logs.append(f"      O {oi} has no H within cutoffs")
    candidates = [(oi, hi, hd) for (oi, hi, hd) in oh_info if hi is not None]
    if not candidates:
        logs.append("    No O–H found in this group; assuming already deprotonated.")
        return mol_struct, O_group, None, None
    oi_star, h_star, d_star = sorted(candidates, key=lambda t: t[2])[0]
    logs.append(f"    Deprotonation: removing H {h_star} from O {oi_star} (d={d_star:.3f} Å)")
    for r in sorted([h_star], reverse=True):
        mol_struct.remove_sites([r])
    O_group_new = remap_after_deletions([h_star], O_group)
    oi_star_new = remap_after_deletions([h_star], [oi_star])[0]
    return mol_struct, O_group_new, oi_star_new, h_star



def _group_outward_score(coords: np.ndarray, O_group: List[int]) -> float:
    mol_min = coords.min(axis=0)
    mol_max = coords.max(axis=0)
    center_xy = (mol_min[:2] + mol_max[:2]) / 2
    return float(sum(np.linalg.norm(coords[oi][:2] - center_xy) for oi in O_group) / len(O_group))


# ============================================================
# Family-specific group detectors
# ============================================================
def find_phosphonic_candidates(structure: Structure):
    coords = np.array(structure.cart_coords)
    species = [s.species_string for s in structure.sites]
    P_idx = [i for i, e in enumerate(species) if e == "P"]
    O_idx = [i for i, e in enumerate(species) if e == "O"]
    cands = []
    for pi in P_idx:
        neigh = []
        for oi in O_idx:
            d = np.linalg.norm(coords[oi] - coords[pi])
            if 1.3 <= d <= 1.9:
                neigh.append((oi, d))
        neigh.sort(key=lambda x: x[1])
        O3 = [oi for oi, _ in neigh[:3]]
        if len(O3) < 3:
            continue
        cands.append({
            "family": "PO3",
            "center_idx": pi,
            "center_label": "P",
            "O_group": O3,
            "score": _group_outward_score(coords, O3),
        })
    cands.sort(key=lambda d: d["score"], reverse=True)
    return cands



def find_sulfonic_candidates(structure: Structure):
    coords = np.array(structure.cart_coords)
    species = [s.species_string for s in structure.sites]
    S_idx = [i for i, e in enumerate(species) if e == "S"]
    O_idx = [i for i, e in enumerate(species) if e == "O"]
    cands = []
    for si in S_idx:
        neigh = []
        for oi in O_idx:
            d = np.linalg.norm(coords[oi] - coords[si])
            if 1.3 <= d <= 1.9:
                neigh.append((oi, d))
        neigh.sort(key=lambda x: x[1])
        O3 = [oi for oi, _ in neigh[:3]]
        if len(O3) < 3:
            continue
        cands.append({
            "family": "SO3",
            "center_idx": si,
            "center_label": "S",
            "O_group": O3,
            "score": _group_outward_score(coords, O3),
        })
    cands.sort(key=lambda d: d["score"], reverse=True)
    return cands



def find_carboxylic_candidates(structure: Structure):
    coords = np.array(structure.cart_coords)
    species = [s.species_string for s in structure.sites]
    C_idx = [i for i, e in enumerate(species) if e == "C"]
    O_idx = [i for i, e in enumerate(species) if e == "O"]
    cands = []
    for ci in C_idx:
        neigh = []
        for oi in O_idx:
            d = np.linalg.norm(coords[oi] - coords[ci])
            if 1.15 <= d <= 1.45:
                neigh.append((oi, d))
        neigh.sort(key=lambda x: x[1])
        O2 = [oi for oi, _ in neigh[:2]]
        if len(O2) < 2:
            continue
        cands.append({
            "family": "COOH",
            "center_idx": ci,
            "center_label": "C",
            "O_group": O2,
            "score": _group_outward_score(coords, O2),
        })
    cands.sort(key=lambda d: d["score"], reverse=True)
    return cands



def get_candidates_by_mode(structure: Structure, mode: str):
    mode = mode.upper()
    if mode == "PO3":
        return find_phosphonic_candidates(structure)
    if mode == "SO3":
        return find_sulfonic_candidates(structure)
    if mode == "COOH":
        return find_carboxylic_candidates(structure)
    if mode == "AUTO":
        cands = []
        cands.extend(find_phosphonic_candidates(structure))
        cands.extend(find_sulfonic_candidates(structure))
        cands.extend(find_carboxylic_candidates(structure))
        family_rank = {"PO3": 0, "SO3": 1, "COOH": 2}
        cands.sort(key=lambda d: (family_rank[d["family"]], -d["score"]))
        return cands
    raise ValueError(f"Unsupported mode: {mode}")


# ============================================================
# Placement and randomized exhaustive search
# ============================================================
def place_with_anchor(
    nc_struct: Structure,
    mol_src: Structure,
    anchor_O: int,
    other_O: int,
    center_idx: int,
    pb_vacancy_site: np.ndarray,
    n_substrate: int,
    center_label: str,
    cs_targets: List[np.ndarray],
    logs: List[str],
):
    mol_loc = copy.deepcopy(mol_src)
    anchor_pos = np.array(mol_loc.cart_coords)[anchor_O]
    for s in mol_loc.sites:
        s.coords = np.array(s.coords) - anchor_pos

    center_pos = np.array(mol_loc.cart_coords)[center_idx]
    v_OC = center_pos

    asf = AdsorbateSiteFinder(nc_struct, height=1.0)
    ads_sites = asf.find_adsorption_sites()
    safe_site = min(ads_sites["all"], key=lambda s: np.linalg.norm(np.array(s) - pb_vacancy_site))
    safe_site = np.array(safe_site)
    top_surface_z = max(site.coords[2] for site in nc_struct.sites)
    if safe_site[2] < top_surface_z + 1.0:
        safe_site[2] = top_surface_z + 1.0

    v_pb = pb_vacancy_site - safe_site
    if np.linalg.norm(v_OC) < 1e-9 or np.linalg.norm(v_pb) < 1e-9:
        R = np.eye(3)
    else:
        R = rotation_matrix_from_vectors(v_OC, -v_pb)

    rot_coords = np.array([R @ np.array(p) for p in mol_loc.cart_coords]) + safe_site
    ads_species = [s.species_string for s in mol_loc.sites]
    adsorbate = Molecule(ads_species, rot_coords)
    combined = asf.add_adsorbate(adsorbate, [0, 0, 0], translate=False, reorient=False)

    anchor_abs = n_substrate + anchor_O
    other_abs = n_substrate + other_O
    anchor_xyz = np.array(combined.sites[anchor_abs].coords)
    d_pb = np.linalg.norm(anchor_xyz - pb_vacancy_site)
    logs.append(f"    Pb–O pre-adjust distance: {d_pb:.3f} Å")

    if abs(d_pb - PB_O_TARGET_DIST) > 1e-6:
        dir_vec = anchor_xyz - pb_vacancy_site
        nrm = np.linalg.norm(dir_vec)
        if nrm < 1e-12:
            dir_vec = np.array([0.0, 0.0, 1.0])
            nrm = 1.0
        unit = dir_vec / nrm
        new_anchor = pb_vacancy_site + PB_O_TARGET_DIST * unit
        delta = new_anchor - anchor_xyz
        for i in range(n_substrate, len(combined.sites)):
            combined.sites[i].coords = np.array(combined.sites[i].coords) + delta
        anchor_xyz = np.array(combined.sites[anchor_abs].coords)
        d_pb = np.linalg.norm(anchor_xyz - pb_vacancy_site)

    logs.append(
        f"    Pb–O post-adjust distance: {d_pb:.3f} Å [{center_label}-centered placement; "
        f"target distance {PB_O_TARGET_DIST:.2f} Å]"
    )

    ref_points = [pb_vacancy_site] + list(cs_targets)
    ref_labels = ["Pb_vacancy"] + [f"Cs[{i}]" for i in range(len(cs_targets))]
    ref_per_min = per_point_min_distance_to_any_ligand_atom(combined, n_substrate, ref_points)
    ref_min, ref_pair = min_distance_points_to_any_ligand_atom(combined, n_substrate, ref_points)
    logs.append(
        "    Initial min(Pb/Cs reference point ↔ any ligand atom): "
        + ", ".join(f"{ref_labels[idx]}={d:.3f} Å" for idx, d, _ in ref_per_min)
    )
    if ref_pair[0] is not None:
        logs.append(
            f"    Initial overall min(Pb/Cs reference point ↔ any ligand atom) = {ref_min:.3f} Å "
            f"[{ref_labels[ref_pair[0]]} to ligand atom index {ref_pair[1]}]; "
            f"required ≥ {REFERENCE_POINT_MIN_DIST:.1f} Å"
        )

    ok_cs_target, cs_target_min, _margin, cs_target_pair, cs_target_per = check_cs_target_thresholds(
        combined, n_substrate, cs_targets
    )
    logs.append(
        "    Initial min(Cs target ↔ ligand atom): "
        + ", ".join(f"Cs[{idx}]={d:.3f} Å" for idx, d, _ in cs_target_per)
    )
    if cs_target_pair[0] is not None:
        req = cs_target_pair[2]
        logs.append(
            f"    Initial overall min(Cs target ↔ ligand atom) = {cs_target_min:.3f} Å "
            f"[Cs[{cs_target_pair[0]}] to ligand atom index {cs_target_pair[1]}]; "
            f"required > {req:.1f} Å"
        )

    return combined, anchor_abs, other_abs



def structure_signature(structure: Structure, n_substrate: int, decimals: int = 2):
    species = tuple(site.species_string for site in structure.sites[n_substrate:])
    coords = np.round(np.array(structure.cart_coords[n_substrate:]), decimals=decimals)
    return (species, tuple(map(tuple, coords)))



def evaluate_pose(
    structure: Structure,
    n_substrate: int,
    anchor_abs_idx: int,
    other_abs_idx: int,
    cs_targets: List[np.ndarray],
    pb_vacancy_site: np.ndarray,
):
    ref_points = [pb_vacancy_site] + list(cs_targets)

    pb_o_now = np.linalg.norm(np.array(structure.sites[anchor_abs_idx].coords) - pb_vacancy_site)
    ob_new = np.array(structure.sites[other_abs_idx].coords)
    dlist = [np.linalg.norm(ob_new - ct) for ct in cs_targets]
    min_cs = min(dlist)
    ok_cs_window = (CS_D_MIN <= min_cs <= CS_D_MAX)

    ok_gen, dmin_gen, *_ = check_ads_sub_thresholds(
        structure, n_substrate, threshold_H=GENERAL_H_MIN_DIST, threshold_other=GENERAL_OTHER_MIN_DIST
    )
    intra = check_intramolecular_overlaps(structure, n_substrate, factor=0.8)

    ok_pair, d_pair, pair_margin, pair_info = check_substrate_cs_thresholds(structure, n_substrate)

    ok_cs_target, d_cs_target, target_margin, target_info, _target_per = check_cs_target_thresholds(
        structure, n_substrate, cs_targets
    )

    d_ref_any, ref_pair = min_distance_points_to_any_ligand_atom(structure, n_substrate, ref_points)
    ok_ref_any = d_ref_any >= REFERENCE_POINT_MIN_DIST

    ok_pb_o = abs(pb_o_now - PB_O_TARGET_DIST) <= 0.05

    valid = ok_pb_o and ok_cs_window and ok_gen and (not intra) and ok_pair and ok_cs_target and ok_ref_any
    score = (
        abs(min_cs - CS_D_IDEAL),
        abs(pb_o_now - PB_O_TARGET_DIST),
        -pair_margin,
        -target_margin,
        -d_ref_any,
        -dmin_gen,
    )

    return {
        "valid": valid,
        "score": score,
        "pb_o": pb_o_now,
        "min_cs": min_cs,
        "dmin_gen": dmin_gen,
        "d_pair": d_pair,
        "pair_margin": pair_margin,
        "d_cs_target": d_cs_target,
        "target_margin": target_margin,
        "d_ref_any": d_ref_any,
        "dlist": dlist,
        "pair_info": pair_info,
        "target_info": target_info,
        "ref_pair": ref_pair,
    }



def search_orientations_about_anchor(
    S0: Structure,
    n_substrate: int,
    anchor_abs_idx: int,
    other_abs_idx: int,
    cs_targets: List[np.ndarray],
    pb_vacancy_site: np.ndarray,
    logs: List[str],
    max_configs: int = MAX_CONFIGS_PER_LIGAND,
):
    logs.append(f"    [OVERLAP RULE] General substrate–ligand cutoff: H ≥ {GENERAL_H_MIN_DIST:.1f} Å, others ≥ {GENERAL_OTHER_MIN_DIST:.1f} Å")
    logs.append(f"    [OVERLAP RULE] All substrate Cs vs ligand atoms: H > {CS_H_MIN_DIST:.1f} Å, non-H > {CS_OTHER_MIN_DIST:.1f} Å")
    logs.append(f"    [OVERLAP RULE] The 3 specified Cs targets vs ligand atoms: H > {CS_H_MIN_DIST:.1f} Å, non-H > {CS_OTHER_MIN_DIST:.1f} Å")
    logs.append(f"    [OVERLAP RULE] For Pb-vacancy + 3 Cs reference points, min (reference point ↔ any ligand atom) ≥ {REFERENCE_POINT_MIN_DIST:.1f} Å")
    logs.append(
        f"    Building all combinations for randomized exhaustive search: "
        f"Pb–O fixed at {PB_O_TARGET_DIST:.2f} Å, "
        f"θ = 0–180° (step {THETA_STEP_DEG}°), φ = 0–350° (step {PHI_STEP_DEG}°)."
    )
    logs.append(
        f"    All combinations are generated first and then evaluated in randomized order with fixed seed {RANDOM_SEARCH_SEED}."
    )

    oa0 = np.array(S0.sites[anchor_abs_idx].coords)
    ob0 = np.array(S0.sites[other_abs_idx].coords)
    v = ob0 - oa0
    L = np.linalg.norm(v)
    v_hat = v / (L + 1e-15)

    combos = [
        (int(theta_deg), int(phi_deg))
        for theta_deg in range(0, 181, THETA_STEP_DEG)
        for phi_deg in range(0, 360, PHI_STEP_DEG)
    ]

    rng = random.Random(RANDOM_SEARCH_SEED)
    rng.shuffle(combos)

    valid_pose_records = []
    near_pose_records = []
    seen_valid = set()
    seen_near = set()

    for theta_deg, phi_deg in combos:
        translated = S0.copy()
        current_anchor = np.array(translated.sites[anchor_abs_idx].coords)
        dir_vec = current_anchor - pb_vacancy_site
        nrm = np.linalg.norm(dir_vec)
        if nrm < 1e-12:
            dir_vec = np.array([0.0, 0.0, 1.0], dtype=float)
            nrm = 1.0
        anchor_target = pb_vacancy_site + PB_O_TARGET_DIST * (dir_vec / nrm)
        delta = anchor_target - current_anchor
        for idx in range(n_substrate, len(translated.sites)):
            translated.sites[idx].coords = np.array(translated.sites[idx].coords) + delta

        oa = np.array(translated.sites[anchor_abs_idx].coords)

        theta = math.radians(theta_deg)
        phi = math.radians(phi_deg)
        w_hat = np.array([
            math.sin(theta) * math.cos(phi),
            math.sin(theta) * math.sin(phi),
            math.cos(theta),
        ], dtype=float)
        if np.linalg.norm(w_hat) < 1e-12:
            continue
        w_hat /= np.linalg.norm(w_hat)
        Rmap = rotation_matrix_from_vectors(v_hat, w_hat)

        S = translated.copy()
        for idx in range(n_substrate, len(S.sites)):
            p = np.array(translated.sites[idx].coords)
            S.sites[idx].coords = oa + Rmap @ (p - oa)

        metrics = evaluate_pose(S, n_substrate, anchor_abs_idx, other_abs_idx, cs_targets, pb_vacancy_site)
        sig = structure_signature(S, n_substrate, decimals=2)
        record = {
            "structure": S,
            "theta": theta_deg,
            "phi": phi_deg,
            **metrics,
        }
        if metrics["valid"]:
            if sig not in seen_valid:
                valid_pose_records.append(record)
                seen_valid.add(sig)
        else:
            if sig not in seen_near:
                near_pose_records.append(record)
                seen_near.add(sig)

    valid_pose_records.sort(key=lambda r: r["score"])
    near_pose_records.sort(key=lambda r: r["score"])

    logs.append(f"    Randomized exhaustive search completed: checked {len(combos)} pose combinations.")
    logs.append(f"    Unique valid poses found: {len(valid_pose_records)}")

    if valid_pose_records:
        top_valid = valid_pose_records[:max_configs]
        best = top_valid[0]
        logs.append(
            f"    Best valid pose: Pb–O={best['pb_o']:.3f} Å, θ={best['theta']}°, φ={best['phi']}° | "
            f"min Cs–O_other={best['min_cs']:.3f} Å | min_clear={best['dmin_gen']:.3f} Å | "
            f"substrate-Cs min={best['d_pair']:.3f} Å | target-Cs min={best['d_cs_target']:.3f} Å"
        )
        return top_valid, True

    if near_pose_records:
        best = near_pose_records[0]
        logs.append(
            f"    No fully valid pose found. Returning the best near-match only: "
            f"Pb–O={best['pb_o']:.3f} Å, θ={best['theta']}°, φ={best['phi']}° | "
            f"min Cs–O_other={best['min_cs']:.3f} Å | min_clear={best['dmin_gen']:.3f} Å | "
            f"substrate-Cs min={best['d_pair']:.3f} Å | target-Cs min={best['d_cs_target']:.3f} Å"
        )
        return [best], False

    logs.append("    No pose could be generated. Returning the initial placed pose only.")
    fallback = {
        "structure": S0,
        "theta": None,
        "phi": None,
        **evaluate_pose(S0, n_substrate, anchor_abs_idx, other_abs_idx, cs_targets, pb_vacancy_site),
    }
    return [fallback], False


# ============================================================
# Attachment workflow
# ============================================================
def attach_ligand_configs(
    nc_struct: Structure,
    ligand_struct: Structure,
    pb_vacancy_site: np.ndarray,
    cs_targets: List[np.ndarray],
    mode: str,
    max_configs: int = MAX_CONFIGS_PER_LIGAND,
):
    logs = []
    n_substrate = len(nc_struct.sites)

    logs.append("=== Substrate ===")
    logs.append(f"Pb-vacancy target: {pb_vacancy_site.tolist()}")
    logs.append(f"# substrate sites: {n_substrate}")
    logs.append("=== Cs targets ===")
    for i, cs in enumerate(cs_targets):
        logs.append(f"Cs[{i}] = {cs.tolist()}")
    logs.append(f"Ligand family mode: {mode}")

    candidates = get_candidates_by_mode(ligand_struct, mode)
    if not candidates:
        raise RuntimeError(f"No valid ligand group found for mode={mode}.")

    pose_pool = []

    for idx, cand in enumerate(candidates, start=1):
        family = cand["family"]
        center_idx = cand["center_idx"]
        center_label = cand["center_label"]
        O_group = cand["O_group"]

        logs.append("")
        logs.append(f"--- Candidate {idx} [{family}] ---")
        logs.append(f"{center_label} index = {center_idx}, O-group = {O_group}")

        mol_work = copy.deepcopy(ligand_struct)
        mol_work, O_group_new, O_deprot, _ = remove_one_H_from_group(mol_work, O_group, logs)

        if len(O_group_new) < 2:
            logs.append("  Rejected: not enough O atoms remain after deprotonation.")
            continue

        anchor_choices = []
        if O_deprot is not None:
            anchor_choices.append(O_deprot)
        else:
            anchor_choices.extend(O_group_new)

        local_pose_count = 0
        seen_anchor_other = set()
        for O_anchor in anchor_choices:
            other_candidates = [o for o in O_group_new if o != O_anchor]
            for O_other in other_candidates:
                if (O_anchor, O_other) in seen_anchor_other:
                    continue
                seen_anchor_other.add((O_anchor, O_other))

                logs.append(f"Using O_anchor={O_anchor} (binds Pb), O_other={O_other}")

                combined, Oa_abs, Ob_abs = place_with_anchor(
                    nc_struct=nc_struct,
                    mol_src=mol_work,
                    anchor_O=O_anchor,
                    other_O=O_other,
                    center_idx=center_idx,
                    pb_vacancy_site=pb_vacancy_site,
                    n_substrate=n_substrate,
                    center_label=center_label,
                    cs_targets=cs_targets,
                    logs=logs,
                )

                ob0 = np.array(combined.sites[Ob_abs].coords)
                base_dists = [np.linalg.norm(ob0 - cs) for cs in cs_targets]
                logs.append("Base O_other distances to Cs targets: " + ", ".join(f"{d:.3f}" for d in base_dists))

                pose_records, _had_valid = search_orientations_about_anchor(
                    combined,
                    n_substrate,
                    Oa_abs,
                    Ob_abs,
                    cs_targets,
                    pb_vacancy_site,
                    logs,
                    max_configs=max_configs,
                )

                for record in pose_records:
                    pose_pool.append({
                        "family": family,
                        "structure": record["structure"],
                        "deprotonated_ligand": copy.deepcopy(mol_work),
                        "metrics": record,
                        "anchor_O": O_anchor,
                        "other_O": O_other,
                        "candidate_index": idx,
                    })
                    local_pose_count += 1

        logs.append(f"  Candidate {idx} contributed {local_pose_count} pose(s) to the global pool.")

    if not pose_pool:
        raise RuntimeError("Failed to generate any candidate placement.")

    pose_pool.sort(key=lambda p: (not p["metrics"]["valid"], p["metrics"]["score"]))

    unique_pose_pool = []
    seen = set()
    for pose in pose_pool:
        sig = structure_signature(pose["structure"], n_substrate, decimals=2)
        if sig in seen:
            continue
        seen.add(sig)
        unique_pose_pool.append(pose)

    selected = unique_pose_pool[:max_configs]
    if not selected:
        selected = [unique_pose_pool[0]]

    logs.append("")
    logs.append(f"Selected {len(selected)} configuration(s) for downstream output/relaxation (maximum allowed = {max_configs}).")
    for i, pose in enumerate(selected, start=1):
        m = pose["metrics"]
        logs.append(
            f"  config{i}: family={pose['family']}, valid={m['valid']}, Pb–O={m['pb_o']:.3f} Å, "
            f"min Cs–O_other={m['min_cs']:.3f} Å, substrate-Cs min={m['d_pair']:.3f} Å, target-Cs min={m['d_cs_target']:.3f} Å"
        )

    detected_family = selected[0]["family"]
    return selected, detected_family, "\n".join(logs)


# ============================================================
# M3GNet relaxation helpers
# ============================================================
def load_m3gnet_potential_from_zip(zip_path: str):
    import matgl

    tmpdir = tempfile.mkdtemp(prefix="m3gnet_model_")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmpdir)

    candidate_dirs = set()
    for root, _dirs, files in os.walk(tmpdir):
        file_set = set(files)
        if (
            "model.pt" in file_set
            or "state.pt" in file_set
            or "model.json" in file_set
            or "metadata.json" in file_set
        ):
            candidate_dirs.add(root)

    if not candidate_dirs:
        raise RuntimeError(
            "No MatGL model files found in ZIP. Expected model.pt / state.pt / model.json either at ZIP root or inside one folder."
        )

    model_dir = sorted(candidate_dirs, key=len)[0]
    errors = []
    for backend in ("DGL", "PYG"):
        try:
            os.environ["MATGL_BACKEND"] = backend
            if hasattr(matgl, "set_backend"):
                matgl.set_backend(backend)
            potential = matgl.load_model(path=model_dir)
            return potential, backend, model_dir
        except Exception as e:
            errors.append(f"{backend}: {e}")

    raise RuntimeError(
        "Could not load the uploaded MatGL model. Tried backends in order DGL -> PYG. "
        + " | ".join(errors)
    )



def build_matgl_ase_calculator(potential):
    errors = []

    try:
        from matgl.ext.ase import M3GNetCalculator

        kwargs = {"potential": potential}
        try:
            sig = inspect.signature(M3GNetCalculator)
            if "compute_stress" in sig.parameters:
                kwargs["compute_stress"] = False
            if "stress_weight" in sig.parameters:
                kwargs["stress_weight"] = 1.0 / 160.21766208
        except Exception:
            kwargs["compute_stress"] = False
            kwargs["stress_weight"] = 1.0 / 160.21766208

        calc = M3GNetCalculator(**kwargs)
        return calc, "matgl.ext.ase.M3GNetCalculator"
    except Exception as e:
        errors.append(f"matgl.ext.ase.M3GNetCalculator: {e}")

    try:
        from matgl.ext._ase_dgl import PESCalculator

        kwargs = {}
        try:
            sig = inspect.signature(PESCalculator)
            if "compute_stress" in sig.parameters:
                kwargs["compute_stress"] = False
            if "stress_weight" in sig.parameters:
                kwargs["stress_weight"] = 1.0 / 160.21766208
        except Exception:
            pass

        try:
            calc = PESCalculator(potential=potential, **kwargs)
        except TypeError:
            calc = PESCalculator(potential, **kwargs)

        return calc, "matgl.ext._ase_dgl.PESCalculator"
    except Exception as e:
        errors.append(f"matgl.ext._ase_dgl.PESCalculator: {e}")

    raise RuntimeError("Could not create a MatGL ASE calculator. Tried " + " | ".join(errors))



def relax_with_m3gnet(struct: Structure, potential, fmax: float, steps: int):
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.optimize.fire import FIRE

    adaptor = AseAtomsAdaptor()
    errors = []

    for import_path in ("matgl.ext.ase", "matgl.ext._ase_dgl"):
        try:
            if import_path == "matgl.ext.ase":
                from matgl.ext.ase import Relaxer
            else:
                from matgl.ext._ase_dgl import Relaxer

            try:
                relaxer = Relaxer(potential=potential, optimizer="FIRE", relax_cell=False)
            except TypeError:
                try:
                    relaxer = Relaxer(potential=potential, relax_cell=False)
                except TypeError:
                    relaxer = Relaxer(potential=potential)

            results = relaxer.relax(struct, fmax=float(fmax), steps=int(steps), verbose=False)

            final_struct = None
            energy = None

            if isinstance(results, dict):
                final_struct = results.get("final_structure", None)
                if final_struct is None:
                    final_atoms = results.get("final_atoms", None)
                    if final_atoms is not None:
                        final_struct = adaptor.get_structure(final_atoms)

                traj = results.get("trajectory", None)
                if traj is not None:
                    if hasattr(traj, "energies") and len(traj.energies) > 0:
                        energy = float(traj.energies[-1])
                    elif isinstance(traj, list) and len(traj) > 0:
                        last = traj[-1]
                        if hasattr(last, "get_potential_energy"):
                            energy = float(last.get_potential_energy())

                if energy is None and results.get("energy", None) is not None:
                    energy = float(results["energy"])

            elif isinstance(results, tuple):
                if len(results) >= 1:
                    obj0 = results[0]
                    if isinstance(obj0, Structure):
                        final_struct = obj0
                    else:
                        try:
                            final_struct = adaptor.get_structure(obj0)
                        except Exception:
                            final_struct = None
                if len(results) >= 2:
                    obj1 = results[1]
                    if hasattr(obj1, "energies") and len(obj1.energies) > 0:
                        energy = float(obj1.energies[-1])
                    else:
                        try:
                            energy = float(obj1)
                        except Exception:
                            pass

            if final_struct is None:
                raise RuntimeError(f"Unexpected Relaxer return type from {import_path}: {type(results)}")

            if energy is None:
                atoms = adaptor.get_atoms(final_struct)
                calc, _ = build_matgl_ase_calculator(potential)
                atoms.calc = calc
                energy = float(atoms.get_potential_energy())

            return final_struct, energy

        except Exception as e:
            errors.append(f"{import_path}.Relaxer: {e}")

    try:
        atoms = adaptor.get_atoms(struct)
        calc, _ = build_matgl_ase_calculator(potential)
        atoms.calc = calc
        opt = FIRE(atoms, logfile=None)
        opt.run(fmax=float(fmax), steps=int(steps))
        final_struct = adaptor.get_structure(atoms)
        energy = float(atoms.get_potential_energy())
        return final_struct, energy
    except Exception as e:
        errors.append(f"ASE FIRE fallback: {e}")

    raise RuntimeError("Relaxation failed. Tried: " + " | ".join(errors))


# ============================================================
# File helpers
# ============================================================
def save_structure(structure: Structure, path: str):
    Poscar(structure).write_file(path)



def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)



def create_zip_from_folder(folder_path: str, zip_prefix: str) -> str:
    fd, zip_path = tempfile.mkstemp(prefix=zip_prefix, suffix=".zip")
    os.close(fd)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(folder_path):
            for file in files:
                full = os.path.join(root, file)
                rel = os.path.relpath(full, folder_path)
                zf.write(full, arcname=rel)
    return zip_path



def ligand_stem_from_path(path: str) -> str:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in stem)
    return safe or "ligand"



def collect_input_ligand_paths(single_ligand_file, ligand_batch_files):
    paths = []
    temp_extract_dirs = []

    if single_ligand_file:
        paths.append(single_ligand_file)

    batch_items = []
    if ligand_batch_files:
        if isinstance(ligand_batch_files, (list, tuple)):
            batch_items.extend(ligand_batch_files)
        else:
            batch_items.append(ligand_batch_files)

    for item in batch_items:
        ext = os.path.splitext(item)[1].lower()
        if ext == ".zip":
            extract_dir = tempfile.mkdtemp(prefix="ligand_batch_")
            temp_extract_dirs.append(extract_dir)
            with zipfile.ZipFile(item, "r") as zf:
                zf.extractall(extract_dir)
            for root, _dirs, files in os.walk(extract_dir):
                for file in files:
                    fext = os.path.splitext(file)[1].lower()
                    if fext in SUPPORTED_EXTENSIONS:
                        paths.append(os.path.join(root, file))
        elif ext in SUPPORTED_EXTENSIONS:
            paths.append(item)

    deduped = []
    seen = set()
    for p in paths:
        rp = os.path.abspath(p)
        if rp not in seen:
            deduped.append(p)
            seen.add(rp)

    return deduped, temp_extract_dirs



def write_summary_csv(rows: List[dict]) -> str:
    fd, csv_path = tempfile.mkstemp(prefix="summary_", suffix=".csv")
    os.close(fd)
    fieldnames = [
        "ligand_name",
        "detected_family",
        "n_attached_configs",
        "best_relaxed_config",
        "E_complex_best_eV",
        "E_ligand_neutral_eV",
        "E_ads_best_eV",
        "valid_configs_found",
        "status",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


# ============================================================
# Main app workflow
# ============================================================
def run_app(
    nc_file,
    ligand_file,
    ligand_batch_files,
    model_zip_file,
    family_mode,
    E_pristine,
    fmax,
    steps,
    do_relax,
    pb_x, pb_y, pb_z,
    cs1_x, cs1_y, cs1_z,
    cs2_x, cs2_y, cs2_z,
    cs3_x, cs3_y, cs3_z,
):
    status_lines = []
    final_complex_energy = None
    final_ligand_energy = None
    final_ads_energy = None
    summary_csv = None
    attached_zip = None
    relaxed_zip = None

    def pack():
        return (
            "\n".join(status_lines),
            final_complex_energy,
            final_ligand_energy,
            final_ads_energy,
            summary_csv,
            attached_zip,
            relaxed_zip,
        )

    if nc_file is None:
        status_lines.append("Please upload the nanocrystal file.")
        yield pack()
        return

    if ligand_file is None and not ligand_batch_files:
        status_lines.append("Please upload either one ligand structure file or multiple ligand files / a ZIP archive.")
        yield pack()
        return

    try:
        nc_struct = Structure.from_file(nc_file)
    except Exception as e:
        status_lines.append(f"Failed to read nanocrystal file: {e}")
        yield pack()
        return

    try:
        pb_vacancy_site = parse_xyz_triplet(pb_x, pb_y, pb_z)
        cs_targets = [
            parse_xyz_triplet(cs1_x, cs1_y, cs1_z),
            parse_xyz_triplet(cs2_x, cs2_y, cs2_z),
            parse_xyz_triplet(cs3_x, cs3_y, cs3_z),
        ]
    except Exception as e:
        status_lines.append(f"Failed to parse Pb/Cs coordinates: {e}")
        yield pack()
        return

    try:
        ligand_paths, _temp_dirs = collect_input_ligand_paths(ligand_file, ligand_batch_files)
    except Exception as e:
        status_lines.append(f"Failed to prepare ligand inputs: {e}")
        yield pack()
        return

    if not ligand_paths:
        status_lines.append("No readable ligand structure files were found. Use .vasp/.cif files directly or a ZIP containing them.")
        yield pack()
        return

    status_lines.append(f"Loaded {len(ligand_paths)} ligand structure(s).")
    status_lines.append(f"Search mode = {family_mode}; maximum saved attached configurations per ligand = {MAX_CONFIGS_PER_LIGAND}")
    status_lines.append(f"Specific Cs thresholds: H > {CS_H_MIN_DIST:.1f} Å, non-H > {CS_OTHER_MIN_DIST:.1f} Å")
    status_lines.append("")
    yield pack()

    attached_root = tempfile.mkdtemp(prefix="attached_configs_")
    relaxed_root = tempfile.mkdtemp(prefix="relaxed_best_")
    summary_rows = []

    potential = None
    if do_relax:
        if model_zip_file is None:
            status_lines.append("No M3GNet model ZIP uploaded. Relaxation will be skipped for all ligands.")
            do_relax = False
            yield pack()
        else:
            status_lines.append("Loading uploaded M3GNet model ZIP...")
            yield pack()
            try:
                potential, backend_used, model_dir = load_m3gnet_potential_from_zip(model_zip_file)
                status_lines.append(f"M3GNet model loaded successfully. Backend = {backend_used}")
                status_lines.append(f"Detected model directory inside ZIP: {model_dir}")
            except Exception as e:
                status_lines.append(f"Failed to load M3GNet model from ZIP: {e}")
                status_lines.append("Relaxation will be skipped for all ligands.")
                do_relax = False
            yield pack()
    status_lines.append("")
    yield pack()

    for idx, ligand_path in enumerate(ligand_paths, start=1):
        ligand_name = ligand_stem_from_path(ligand_path)
        status_lines.append("=" * 80)
        status_lines.append(f"[{idx}/{len(ligand_paths)}] Processing ligand: {ligand_name}")
        yield pack()

        try:
            ligand_struct = Structure.from_file(ligand_path)
        except Exception as e:
            status_lines.append(f"Failed to read ligand file: {e}")
            summary_rows.append({
                "ligand_name": ligand_name,
                "detected_family": "",
                "n_attached_configs": 0,
                "best_relaxed_config": "",
                "E_complex_best_eV": "",
                "E_ligand_neutral_eV": "",
                "E_ads_best_eV": "",
                "valid_configs_found": 0,
                "status": f"Failed to read ligand: {e}",
            })
            yield pack()
            continue

        try:
            status_lines.append(f"Starting attachment search for {ligand_name}...")
            yield pack()
            selected_configs, detected_family, attach_logs = attach_ligand_configs(
                nc_struct=nc_struct,
                ligand_struct=ligand_struct,
                pb_vacancy_site=pb_vacancy_site,
                cs_targets=cs_targets,
                mode=family_mode,
                max_configs=MAX_CONFIGS_PER_LIGAND,
            )
            status_lines.append(f"Detected/used family: {detected_family}")
            status_lines.append(attach_logs)
            yield pack()
        except Exception as e:
            tb = traceback.format_exc()
            status_lines.append(f"Attachment failed: {e}")
            status_lines.append(tb)
            summary_rows.append({
                "ligand_name": ligand_name,
                "detected_family": "",
                "n_attached_configs": 0,
                "best_relaxed_config": "",
                "E_complex_best_eV": "",
                "E_ligand_neutral_eV": "",
                "E_ads_best_eV": "",
                "valid_configs_found": 0,
                "status": f"Attachment failed: {e}",
            })
            yield pack()
            continue

        ligand_attached_dir = os.path.join(attached_root, ligand_name)
        ensure_dir(ligand_attached_dir)

        valid_configs_found = 0
        for i, pose in enumerate(selected_configs, start=1):
            cfg_name = f"{ligand_name}_NC_config{i}.vasp"
            cfg_path = os.path.join(ligand_attached_dir, cfg_name)
            save_structure(pose["structure"], cfg_path)
            if pose["metrics"]["valid"]:
                valid_configs_found += 1
        status_lines.append(f"Saved {len(selected_configs)} attached configuration(s) for {ligand_name}.")
        yield pack()

        E_ligand_neutral = None
        best_relaxed_name = ""
        best_complex_energy = None
        best_ads_energy = None
        best_relaxed_struct = None
        best_relaxed_idx = None
        best_status = "Attached configurations saved."

        if do_relax and potential is not None:
            status_lines.append(f"Starting neutral ligand relaxation for {ligand_name}...")
            yield pack()
            try:
                _neutral_relaxed_struct, E_ligand_neutral = relax_with_m3gnet(ligand_struct, potential, fmax, steps)
                status_lines.append(f"Neutral ligand relaxation completed for {ligand_name}. E_ligand = {E_ligand_neutral:.6f} eV")
                final_ligand_energy = E_ligand_neutral
            except Exception as e:
                status_lines.append(f"Neutral ligand relaxation failed for {ligand_name}: {type(e).__name__}: {e}")
                E_ligand_neutral = None
            yield pack()

            for i, pose in enumerate(selected_configs, start=1):
                status_lines.append(f"Starting relaxation for {ligand_name} config {i}/{len(selected_configs)}...")
                yield pack()
                try:
                    relaxed_struct, E_complex = relax_with_m3gnet(pose["structure"], potential, fmax, steps)
                    status_lines.append(
                        f"  Config {i} relaxed successfully for {ligand_name}. E_complex = {E_complex:.6f} eV"
                    )
                    if best_complex_energy is None or E_complex < best_complex_energy:
                        best_complex_energy = E_complex
                        best_relaxed_struct = relaxed_struct
                        best_relaxed_idx = i
                        final_complex_energy = E_complex
                except Exception as e:
                    status_lines.append(
                        f"  Config {i} relaxation failed for {ligand_name}: {type(e).__name__}: {e}"
                    )
                yield pack()

            if best_relaxed_struct is not None and best_relaxed_idx is not None:
                ligand_relaxed_dir = os.path.join(relaxed_root, ligand_name)
                ensure_dir(ligand_relaxed_dir)
                best_relaxed_name = f"{ligand_name}_NC_best_config{best_relaxed_idx}_relaxed.vasp"
                best_path = os.path.join(ligand_relaxed_dir, best_relaxed_name)
                save_structure(best_relaxed_struct, best_path)

                if E_ligand_neutral is not None and E_pristine is not None:
                    try:
                        best_ads_energy = float(best_complex_energy) - (
                            float(E_pristine) + float(E_ligand_neutral) - float(HALF_H2_ENERGY)
                        )
                        status_lines.append(
                            f"  Best relaxed configuration for {ligand_name}: config{best_relaxed_idx} | "
                            f"E_complex = {best_complex_energy:.6f} eV | E_ads = {best_ads_energy:.6f} eV"
                        )
                        best_status = "Relaxed successfully; best relaxed structure saved."
                        final_ads_energy = best_ads_energy
                    except Exception as e:
                        status_lines.append(f"  Adsorption energy calculation failed for {ligand_name}: {e}")
                        best_status = "Relaxed, but adsorption energy calculation failed."
                else:
                    status_lines.append(
                        f"  Best relaxed configuration for {ligand_name}: config{best_relaxed_idx} | "
                        f"E_complex = {best_complex_energy:.6f} eV"
                    )
                    best_status = "Relaxed successfully; best relaxed structure saved."
            else:
                best_status = "No configuration relaxed successfully."
            yield pack()
        else:
            best_status = "Relaxation skipped."
            yield pack()

        summary_rows.append({
            "ligand_name": ligand_name,
            "detected_family": detected_family,
            "n_attached_configs": len(selected_configs),
            "best_relaxed_config": best_relaxed_name,
            "E_complex_best_eV": "" if best_complex_energy is None else f"{best_complex_energy:.6f}",
            "E_ligand_neutral_eV": "" if E_ligand_neutral is None else f"{E_ligand_neutral:.6f}",
            "E_ads_best_eV": "" if best_ads_energy is None else f"{best_ads_energy:.6f}",
            "valid_configs_found": valid_configs_found,
            "status": best_status,
        })
        status_lines.append(f"Completed ligand {ligand_name}. Status: {best_status}")
        yield pack()

    summary_csv = write_summary_csv(summary_rows)
    attached_zip = create_zip_from_folder(attached_root, "attached_configs_")
    relaxed_zip = None
    if os.path.isdir(relaxed_root) and any(files for _, _, files in os.walk(relaxed_root)):
        relaxed_zip = create_zip_from_folder(relaxed_root, "relaxed_best_")

    status_lines.append("")
    status_lines.append("Finished processing all ligand inputs.")
    status_lines.append("Downloadable outputs:")
    status_lines.append("1. summary CSV")
    status_lines.append("2. ZIP of all attached NC+ligand configurations")
    status_lines.append("3. ZIP of best relaxed NC+ligand structures (one per ligand, if relaxation succeeded)")
    yield pack()
    return



# ============================================================
# Gradio UI
# ============================================================
GLOBAL_CSS = """
body {
    background-color: #f3f4f6;
}
.gradio-container {
    background-color: #f3f4f6 !important;
    color: #111827;
    font-size: 16px;
}
.gradio-container label {
    color: #111827 !important;
    font-weight: 700 !important;
    font-size: 1.0rem !important;
}
.gradio-container .gr-markdown,
.gradio-container .gr-markdown * {
    color: #111827 !important;
    font-weight: 600;
    font-size: 1.0rem;
}
.gradio-container textarea {
    font-size: 0.95rem;
}
.gradio-container input[type=\"number\"],
.gradio-container input[type=\"text\"] {
    font-size: 0.95rem;
}
"""


with gr.Blocks(title="X-type Ligand Passivation at V_X Sites on Nanocrystal Surfaces") as demo:
    gr.HTML(f"<style>{GLOBAL_CSS}</style>")

    gr.Markdown(
        """
<div style="padding: 1.2rem 1.4rem; border-radius: 16px;
            background: linear-gradient(135deg,#fef9c3,#bbf7d0,#a7f3d0);
            color:#111827;
            box-shadow:0 18px 45px rgba(15,23,42,0.18);">
  <h1 style="margin:0 0 0.45rem 0; font-size: 2.15rem; font-weight: 800; color:#111827;">
    X-type Ligand Passivation at V<sub>X</sub> Sites on Nanocrystal Surfaces
  </h1>
  <p style="margin:0.25rem 0; font-size:1.02rem; font-weight: 700; color:#111827; line-height:1.5;">
    X-type ligand passivation at V<sub>X</sub> defect sites on nanocrystal surfaces, followed by structural optimization using a trained M3GNet machine-learning force field.
  </p>
  <p style="margin-top:0.8rem; font-size:0.98rem; font-weight: 700; color:#111827; line-height:1.6;">
    Kushal Samanta<sup>a,b</sup>, Jyoti Bharti<sup>c</sup>, Arun Mannodi-Kanakkithodi<sup>b*</sup>, Dibyajyoti Ghosh<sup>a,c*</sup><br>
    <span style="font-weight:600;">
    <sup>a</sup> Department of Materials Science and Engineering, Indian Institute of Technology, Delhi-110016, India<br>
    <sup>b</sup> School of Materials Engineering, Purdue University, West Lafayette, IN 47907, United States of America<br>
    <sup>c</sup> Department of Chemistry, Indian Institute of Technology, Delhi-110016, India
    </span>
  </p>
</div>
        """
    )

    gr.Markdown(
        f"""
<div style="margin-top:1.0rem; padding:1.0rem 1.1rem;
            border-radius:12px; background:#ecfdf5;
            border:1px solid #6ee7b7;
            box-shadow:0 4px 12px rgba(15,23,42,0.08);
            font-size:1.0rem; color:#064e3b; font-weight:600; line-height:1.6;">
  <strong>Workflow summary</strong><br><br>
  • Upload the defective nanocrystal structure.<br>
  • Upload either one ligand file, or multiple ligand files / one ZIP archive containing many ligand files.<br>
  • For each ligand, the app generates multiple possible attached NC+ligand configurations from the initial scratch structures, with a maximum of <strong>{MAX_CONFIGS_PER_LIGAND}</strong> saved configurations.<br>
  • The trained M3GNet model evaluates the relaxed structure of the separate ligand molecule as well as the NC–ligand passivated system.<br>
  • The adsorption energy is evaluated as:<br>
    <code>E_ads = E[NC+VCl+ligand] - (E[NC+VCl] + E[ligand] - 1/2 E[H2])</code><br>
    with <code>1/2 E[H2] = {HALF_H2_ENERGY:.6f} eV</code>.<br>
</div>
        """
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("<h3 style='margin-top:1.0rem; color:#1d4ed8; font-size:1.25rem; font-weight:800;'>Input structures</h3>")
            nc_file = gr.File(
                label="Defective nanocrystal (POSCAR / VASP / CIF)",
                type="filepath",
            )
            ligand_file = gr.File(
                label="Optional single ligand structure (POSCAR / VASP / CIF)",
                type="filepath",
            )
            ligand_batch_files = gr.File(
                label="Optional multiple ligand files or one ZIP containing many ligands",
                type="filepath",
                file_count="multiple",
            )
            model_file = gr.File(
                label="Optional MatGL/M3GNet model ZIP (for relaxation)",
                type="filepath",
            )
            family_mode = gr.Dropdown(
                choices=["AUTO", "PO3", "SO3", "COOH"],
                value="AUTO",
                label="Ligand family mode",
            )

        with gr.Column():
            gr.Markdown("<h3 style='margin-top:1.0rem; color:#7c3aed; font-size:1.25rem; font-weight:800;'>Energetics and relaxation options</h3>")
            E_pristine = gr.Number(
                label="Energy of the defective nanocrystal (eV)",
                value=DEFAULT_E_PRISTINE,
            )
            fmax_in = gr.Number(
                label="Force convergence threshold fmax (eV/Å)",
                value=DEFAULT_FMAX,
            )
            steps_in = gr.Number(
                label="Maximum relaxation steps",
                value=DEFAULT_STEPS,
                precision=0,
            )
            do_relax = gr.Checkbox(
                label="Relax the neutral ligand and all attached configurations with M3GNet",
                value=True,
            )

            gr.Markdown("<h3 style='margin-top:1.0rem; color:#ea580c; font-size:1.25rem; font-weight:800;'>Defect / attachment site (Å)</h3>")
            with gr.Row():
                pb_x = gr.Number(label="Defect site x", value=DEFAULT_PB[0])
                pb_y = gr.Number(label="Defect site y", value=DEFAULT_PB[1])
                pb_z = gr.Number(label="Defect site z", value=DEFAULT_PB[2])

    with gr.Row(visible=False):
        cs1_x = gr.Number(label="Cs1 x", value=DEFAULT_CS[0][0])
        cs1_y = gr.Number(label="Cs1 y", value=DEFAULT_CS[0][1])
        cs1_z = gr.Number(label="Cs1 z", value=DEFAULT_CS[0][2])
    with gr.Row(visible=False):
        cs2_x = gr.Number(label="Cs2 x", value=DEFAULT_CS[1][0])
        cs2_y = gr.Number(label="Cs2 y", value=DEFAULT_CS[1][1])
        cs2_z = gr.Number(label="Cs2 z", value=DEFAULT_CS[1][2])
    with gr.Row(visible=False):
        cs3_x = gr.Number(label="Cs3 x", value=DEFAULT_CS[2][0])
        cs3_y = gr.Number(label="Cs3 y", value=DEFAULT_CS[2][1])
        cs3_z = gr.Number(label="Cs3 z", value=DEFAULT_CS[2][2])

    run_btn = gr.Button("Run batch attachment / relaxation workflow", variant="primary")

    gr.Markdown("<h3 style='margin-top:1.1rem; color:#059669; font-size:1.25rem; font-weight:800;'>Status and outputs</h3>")
    status_box = gr.Textbox(label="Status / detailed log", lines=24)

    gr.Markdown("<h3 style='margin-top:1.1rem; color:#7c3aed; font-size:1.25rem; font-weight:800;'>Final energies</h3>")
    with gr.Row():
        final_complex_out = gr.Number(label="Best NC+Ligand M3GNet relaxed energy (eV)")
        final_ligand_out = gr.Number(label="Relaxed neutral ligand energy (eV)")
        final_ads_out = gr.Number(label="Adsorption energy (eV)")

    with gr.Row():
        summary_out = gr.File(label="Summary CSV")
        attached_zip_out = gr.File(label="ZIP of all attached NC+ligand configurations")
        relaxed_zip_out = gr.File(label="ZIP of best relaxed NC+ligand structures")

    run_btn.click(
        fn=run_app,
        inputs=[
            nc_file,
            ligand_file,
            ligand_batch_files,
            model_file,
            family_mode,
            E_pristine,
            fmax_in,
            steps_in,
            do_relax,
            pb_x,
            pb_y,
            pb_z,
            cs1_x,
            cs1_y,
            cs1_z,
            cs2_x,
            cs2_y,
            cs2_z,
            cs3_x,
            cs3_y,
            cs3_z,
        ],
        outputs=[
            status_box,
            final_complex_out,
            final_ligand_out,
            final_ads_out,
            summary_out,
            attached_zip_out,
            relaxed_zip_out,
        ],
    )


demo.queue()

if __name__ == "__main__":
    demo.launch(share=True, debug=True, ssr_mode=False, quiet=True)