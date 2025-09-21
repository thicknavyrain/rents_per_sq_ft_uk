import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    # compute_rental_floor_stats_by_bedrooms.py
    #
    # Compute private-rental floor space stats by bedroom (1,2,3,4+) for the
    # *most recent LAs*, applying post-2023/24 mergers, using:
    #   - EPC microdata (concatenated for merged LAs)
    #   - Precomputed IPF contingency tables (merged dirs for merged LAs)
    #
    # Output:
    #   rental_floor_stats_by_bedrooms.csv  # one row per (merged or standalone) LA
    #
    # Requires:
    #   pip install tqdm

    import os
    import re
    import json
    import math
    import pandas as pd
    import numpy as np
    from tqdm import tqdm

    # -----------------------
    # Config / file locations
    # -----------------------
    AREA_JSON = "area_to_epc_dir.json"   # mapping: Area Name -> EPC subdir
    EPC_ROOT = "EPC_data"                # EPC_data/<EPC_DIR>/certificates.csv
    IPF_ROOT = "ipf_outputs"             # ipf_outputs/<unit_label_sanitized>/<ptype>_contingency.csv

    BEDROOM_BANDS = ["1", "2", "3", "4+"]  # column order
    PTYPES = ["Bungalow", "Flat/Maisonette", "House : Detached", "House : Semi-Detached", "House : Terraced"]

    # ---- Merged LAs → list of historical Area Names (as they appear in ONS/your mapping) ----
    MERGED_LA_GROUPS = {
        "Buckinghamshire": [
            "Aylesbury Vale", "Chiltern", "South Bucks", "Wycombe"
        ],
        "Cumberland": [
            "Allerdale", "Carlisle", "Copeland"
        ],
        "Westmorland and Furness": [
            "Barrow-in-Furness", "Eden", "South Lakeland"
        ],
        "North Yorkshire": [
            "Craven", "Hambleton", "Harrogate", "Richmondshire", "Ryedale", "Scarborough", "Selby"
        ],
        "West Northamptonshire": [
            "Daventry", "South Northamptonshire", "Northampton"
        ],
        "North Northamptonshire": [
            "Corby", "East Northamptonshire", "Kettering", "Wellingborough"
        ],    
    }
    # -----------------------
    # Utilities
    # -----------------------
    def sanitize(name: str) -> str:
        """Make a safe dir/file chunk."""
        name = (name or "").replace("/", "_").replace("\\", "_").replace(":", "").replace("-", "_")
        name = name.replace(" ", "_")
        name = re.sub(r"[^A-Za-z0-9_]+", "", name)
        name = re.sub(r"_+", "_", name).strip("_")
        return name or "unknown"

    def map_epc_to_ons_category(property_type: str, built_form: str) -> str:
        """Apply agreed EPC -> ONS mapping."""
        pt = (property_type or "").strip()
        bf = (built_form or "").strip()
        if pt in {"Flat", "Maisonette"}:
            return "Flat/Maisonette"
        if pt == "Bungalow":
            return "Bungalow"
        if pt == "House":
            if bf == "Detached":
                return "House : Detached"
            if bf == "Semi-Detached":
                return "House : Semi-Detached"
            if bf in {"Mid-Terrace", "End-Terrace", "Enclosed Mid-Terrace", "Enclosed End-Terrace"}:
                return "House : Terraced"
        return "Other/Unknown"

    def nearest_available_H(h: int, available: np.ndarray) -> int:
        """Map H to the nearest available H in the contingency index."""
        idx = np.searchsorted(available, h)
        if idx == 0:
            return int(available[0])
        if idx >= len(available):
            return int(available[-1])
        prev_h, next_h = available[idx-1], available[idx]
        return int(prev_h if abs(h - prev_h) <= abs(next_h - h) else next_h)

    def weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
        """
        Standard weighted mean & std using numpy.average (canonical approach):
          mean = average(values, weights=w)
          var  = average((x-mean)^2, weights=w)
          std  = sqrt(var)
        """
        if values.size == 0 or np.sum(weights) <= 0:
            return (np.nan, np.nan)
        m = np.average(values, weights=weights)
        v = np.average((values - m) ** 2, weights=weights)
        return float(m), float(np.sqrt(max(v, 0.0)))

    def load_epc_certificates(epc_file: str) -> pd.DataFrame:
        usecols = [
            "PROPERTY_TYPE",
            "BUILT_FORM",
            "LOCAL_AUTHORITY",
            "NUMBER_HABITABLE_ROOMS",
            "TOTAL_FLOOR_AREA",
            "TENURE",
        ]
        df = pd.read_csv(epc_file, usecols=usecols, low_memory=False)

        # Normalise strings, coerce numerics
        df["PROPERTY_TYPE"] = df["PROPERTY_TYPE"].astype(str).str.strip()
        df["BUILT_FORM"] = df["BUILT_FORM"].astype(str).str.strip()
        df["TENURE"] = df["TENURE"].astype(str).str.strip()
        df["NUMBER_HABITABLE_ROOMS"] = pd.to_numeric(df["NUMBER_HABITABLE_ROOMS"], errors="coerce")
        df["TOTAL_FLOOR_AREA"] = pd.to_numeric(df["TOTAL_FLOOR_AREA"], errors="coerce")

        # Filter private rentals
        tenure_lc = df["TENURE"].str.casefold()
        is_prs = tenure_lc.str.contains(r"rented\s*\(private\)", na=False) | tenure_lc.str.contains(r"rental\s*\(private\)", na=False)
        df = df[is_prs].copy()

        # Keep valid H and strictly positive floor areas
        df = df.dropna(subset=["NUMBER_HABITABLE_ROOMS", "TOTAL_FLOOR_AREA"])
        df = df[df["TOTAL_FLOOR_AREA"] > 0]

        # H + ons category
        df["H"] = df["NUMBER_HABITABLE_ROOMS"].astype(int).clip(lower=1)
        df["ONS_PTYPE"] = df.apply(lambda r: map_epc_to_ons_category(r["PROPERTY_TYPE"], r["BUILT_FORM"]), axis=1)

        # Keep only target types
        df = df[df["ONS_PTYPE"].isin(PTYPES)]
        return df

    def build_processing_units(area_map: dict[str, str]) -> list[dict]:
        """
        Return a list of processing units for output rows (most recent LAs).
        Each unit:
          {
            "label": <merged LA label OR epc_dir>,
            "epc_dirs": [one or many EPC dirs],
            "ipf_dir": <IPF directory name to read from (sanitized label)>,
            "is_merged": True/False
          }
        Merged units come first; remaining EPC dirs are output as-is.
        """
        from collections import defaultdict
        # Invert mapping: epc_dir -> [area names]
        epc_to_areas = defaultdict(list)
        for area, epc_dir in area_map.items():
            epc_to_areas[epc_dir].append(area)

        used_epc_dirs = set()
        units = []

        # 1) Merged units
        for merged_label, members in MERGED_LA_GROUPS.items():
            members_present = [m for m in members if m in area_map]
            if not members_present:
                continue
            epc_dirs = sorted(set(area_map[m] for m in members_present))
            if not epc_dirs:
                continue
            units.append({
                "label": merged_label,
                "epc_dirs": epc_dirs,
                "ipf_dir": sanitize(merged_label),  # IPF script saved merged outputs under merged label
                "is_merged": True
            })
            used_epc_dirs.update(epc_dirs)

        # 2) Remaining single-epc units
        for epc_dir in sorted(epc_to_areas.keys()):
            if epc_dir in used_epc_dirs:
                continue
            units.append({
                "label": epc_dir,
                "epc_dirs": [epc_dir],
                "ipf_dir": sanitize(epc_dir),
                "is_merged": False
            })
        return units

    # -----------------------
    # Load area map
    # -----------------------
    with open(AREA_JSON, "r", encoding="utf-8") as f:
        AREA_MAP = json.load(f)  # {Area Name: EPC_DIR}

    # -----------------------
    # Prepare result frame
    # -----------------------
    cols = ["Area Name"]
    for b in BEDROOM_BANDS:
        label = "4plus" if b == "4+" else b
        cols += [f"mean_{label}", f"std_{label}"]
    rows = []

    # -----------------------
    # Main loop
    # -----------------------
    units = build_processing_units(AREA_MAP)

    for unit in tqdm(units, desc="LAs (merged + standalone)", total=329):
        label = unit["label"]
        epc_dirs = unit["epc_dirs"]
        ipf_dirname = unit["ipf_dir"]

        # --- Load & concat EPC across all dirs in this unit ---
        epc_frames = []
        for ed in epc_dirs:
            epc_file = os.path.join(EPC_ROOT, ed, "certificates.csv")
            if not os.path.isfile(epc_file):
                print(f"SKIP: EPC file not found for {label} member {ed} at {epc_file}")
                continue
            try:
                epc_frames.append(load_epc_certificates(epc_file))
            except Exception as e:
                print(f"SKIP: failed to load EPC for member {ed}: {e}")

        if not epc_frames:
            # No EPC rows to compute from
            out = {"Area Name": label}
            for b in BEDROOM_BANDS:
                lab = "4plus" if b == "4+" else b
                out[f"mean_{lab}"], out[f"std_{lab}"] = (np.nan, np.nan)
            rows.append(out)
            continue

        df_prs = pd.concat(epc_frames, ignore_index=True)
        if df_prs.empty:
            out = {"Area Name": label}
            for b in BEDROOM_BANDS:
                lab = "4plus" if b == "4+" else b
                out[f"mean_{lab}"], out[f"std_{lab}"] = (np.nan, np.nan)
            rows.append(out)
            continue

        # --- Load contingency tables for this unit (merged or standalone) ---
        ipf_dir = os.path.join(IPF_ROOT, ipf_dirname)
        cont_tables = {}
        for ptype in PTYPES:
            fname = f"{sanitize(ptype)}_contingency.csv"
            path = os.path.join(ipf_dir, fname)
            if os.path.isfile(path):
                mat = pd.read_csv(path, index_col=0)
                mat.columns = [c.strip() for c in mat.columns]
                mat = mat.reindex(columns=BEDROOM_BANDS)              # ensure/order 1,2,3,4+
                mat = mat.apply(pd.to_numeric, errors="coerce")
                mat = mat.dropna(how="all")
                mat.index = pd.to_numeric(mat.index, errors="coerce").astype("Int64")
                mat = mat.dropna(axis=0, how="any")
                mat.index = mat.index.astype(int)
                cont_tables[ptype] = mat

        if not cont_tables:
            print(f"SKIP: No contingency tables found for unit '{label}' in {ipf_dir}")
            out = {"Area Name": label}
            for b in BEDROOM_BANDS:
                lab = "4plus" if b == "4+" else b
                out[f"mean_{lab}"], out[f"std_{lab}"] = (np.nan, np.nan)
            rows.append(out)
            continue

        # --- Aggregate to weighted means/stds by bedroom using H|ptype probabilities ---
        values_by_bed = {b: [] for b in BEDROOM_BANDS}
        weights_by_bed = {b: [] for b in BEDROOM_BANDS}

        for ptype, grp in df_prs.groupby("ONS_PTYPE"):
            mat = cont_tables.get(ptype)
            if mat is None or mat.empty:
                continue
            row_sums = mat[BEDROOM_BANDS].sum(axis=1)
            available_H = np.array(sorted(mat.index.unique()))

            for _, r in grp.iterrows():
                h = int(r["H"])
                y = float(r["TOTAL_FLOOR_AREA"])
                h_use = h if h in mat.index else nearest_available_H(h, available_H)

                probs = mat.loc[h_use, BEDROOM_BANDS].values.astype(float)
                denom = float(row_sums.loc[h_use]) if h_use in row_sums.index else float(np.nansum(probs))
                if not np.isfinite(denom) or denom <= 0:
                    continue
                probs = np.nan_to_num(probs, nan=0.0) / denom

                for b, p in zip(BEDROOM_BANDS, probs):
                    if p <= 0:
                        continue
                    values_by_bed[b].append(y)
                    weights_by_bed[b].append(p)

        # Compute weighted mean/std per bedroom band for this unit
        out = {"Area Name": label}
        for b in BEDROOM_BANDS:
            lab = "4plus" if b == "4+" else b
            vals = np.array(values_by_bed[b], dtype=float)
            wts  = np.array(weights_by_bed[b], dtype=float)
            mean_b, std_b = weighted_mean_std(vals, wts)
            out[f"mean_{lab}"] = mean_b
            out[f"std_{lab}"]  = std_b

        rows.append(out)

    # -----------------------
    # Save result
    # -----------------------
    res = pd.DataFrame(rows, columns=cols)
    res.to_csv("rental_floor_stats_by_bedrooms.csv", index=False)
    print("Saved rental_floor_stats_by_bedrooms.csv")

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
