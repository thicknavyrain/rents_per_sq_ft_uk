import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    # ipf_contingency_tables_all_LAs.py
    #
    # Build HabitableRooms x Bedrooms IPF contingency tables for:
    #  - EVERY EPC directory referenced in area_to_epc_dir.json (with ONS summed across all
    #    Area Names mapped to that directory), AND
    #  - The requested merged authorities (with EPC concatenated across their member EPC dirs
    #    and ONS summed across their member Area Names).
    #
    # Outputs:
    #   ipf_outputs/<unit_label_sanitized>/<ptype>_contingency.csv
    #
    # Requires: pip install ipfn tqdm

    import os
    import re
    import json
    import pandas as pd
    import numpy as np
    from ipfn.ipfn import ipfn as IPF
    from tqdm import tqdm
    from collections import defaultdict

    # -----------------------
    # Config / file locations
    # -----------------------
    AREA_JSON = "area_to_epc_dir.json"     # mapping: Area Name -> EPC subdir
    EPC_ROOT = "EPC_data"
    ONS_DIR = "LA_bedroom_properties"
    OUTPUT_ROOT = "ipf_outputs"

    ONS_FILES = {
        "Bungalow": os.path.join(ONS_DIR, "bungalow_bedrooms.csv"),
        "Flat/Maisonette": os.path.join(ONS_DIR, "flat_maisonette_bedrooms.csv"),
        "House : Detached": os.path.join(ONS_DIR, "house_detached_bedrooms.csv"),
        "House : Semi-Detached": os.path.join(ONS_DIR, "house_semi_detached_bedrooms.csv"),
        "House : Terraced": os.path.join(ONS_DIR, "house_terraced_bedrooms.csv"),
    }

    # ---- Merged LAs → list of historical Area Names (as they appear in ONS files) ----
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
        """Make a safe filename/dirname chunk."""
        name = (name or "").replace("/", "_").replace("\\", "_").replace(":", "").replace("-", "_")
        name = name.replace(" ", "_")
        name = re.sub(r"[^A-Za-z0-9_]+", "", name)
        name = re.sub(r"_+", "_", name).strip("_")
        return name or "unknown"

    def map_epc_to_ons_category(property_type: str, built_form: str) -> str:
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

    def load_epc_certificates(epc_file: str) -> pd.DataFrame:
        usecols = [
            "PROPERTY_TYPE",
            "BUILT_FORM",
            "LOCAL_AUTHORITY",
            "NUMBER_HABITABLE_ROOMS",
            "TOTAL_FLOOR_AREA",
        ]
        df = pd.read_csv(epc_file, usecols=usecols, low_memory=False)
        df["PROPERTY_TYPE"] = df["PROPERTY_TYPE"].astype(str).str.strip()
        df["BUILT_FORM"] = df["BUILT_FORM"].astype(str).str.strip()
        df["LOCAL_AUTHORITY"] = df["LOCAL_AUTHORITY"].astype(str).str.strip()
        df["NUMBER_HABITABLE_ROOMS"] = pd.to_numeric(df["NUMBER_HABITABLE_ROOMS"], errors="coerce")
        df["TOTAL_FLOOR_AREA"] = pd.to_numeric(df["TOTAL_FLOOR_AREA"], errors="coerce")
        df["ONS_PTYPE"] = df.apply(lambda r: map_epc_to_ons_category(r["PROPERTY_TYPE"], r["BUILT_FORM"]), axis=1)
        df = df.dropna(subset=["NUMBER_HABITABLE_ROOMS"])
        df["H"] = df["NUMBER_HABITABLE_ROOMS"].astype(int).clip(lower=1)
        ons_cats = {"Bungalow", "Flat/Maisonette", "House : Detached", "House : Semi-Detached", "House : Terraced"}
        df = df[df["ONS_PTYPE"].isin(ons_cats)]
        return df

    def load_ons_bedroom_counts(ons_file: str, area_name: str) -> pd.Series:
        df = pd.read_csv(ons_file)
        df.columns = [c.strip() for c in df.columns]
        for col in ["1","2","3","4+"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        row = df.loc[df["Area Name"].str.strip().str.casefold() == (area_name or "").strip().casefold()]
        if row.empty:
            raise KeyError(f"Area Name '{area_name}' not found in {ons_file}")
        row = row.iloc[0]
        s = row[["1","2","3","4+"]].astype(float)
        s.index = ["1","2","3","4+"]
        return s

    def sum_ons_for_areas(ons_file: str, area_names: list[str]) -> pd.Series:
        """Sum ONS bedroom counts for a list of Area Names (ignore any not found, warn)."""
        total = pd.Series([0.0, 0.0, 0.0, 0.0], index=["1","2","3","4+"])
        missing = []
        for nm in area_names:
            try:
                s = load_ons_bedroom_counts(ons_file, nm)
                total = total.add(s, fill_value=0.0)
            except Exception:
                missing.append(nm)
        if missing:
            print(f"WARNING: Missing ONS rows in {os.path.basename(ons_file)} for: {', '.join(missing)}")
        return total

    def run_ipf_with_structural_zeros(row_counts: pd.Series, col_counts_scaled: pd.Series) -> pd.DataFrame:
        H_vals = row_counts.index.tolist()
        B_vals = ["1","2","3","4+"]
        m0 = np.ones((len(H_vals), len(B_vals)), dtype=float)
        H_numeric = np.array(H_vals, dtype=int)
        B_numeric = np.array([1,2,3,4], dtype=int)
        mask = (H_numeric[:, None] < B_numeric[None, :])
        m0[mask] = 0.0
        xip = row_counts.values.astype(float)
        xpj = col_counts_scaled.reindex(B_vals).values.astype(float)
        try:
            ipf_engine = IPF(m0.copy(), [xip, xpj], [[0],[1]])
            m_fitted = ipf_engine.iteration()
            row_err = np.abs(m_fitted.sum(axis=1) - xip).max()
            col_err = np.abs(m_fitted.sum(axis=0) - xpj).max()
            if max(row_err, col_err) > 1e-6:
                raise RuntimeError
        except Exception:
            eps = 1e-12
            m0_soft = m0.copy()
            m0_soft[mask] = eps
            ipf_engine = IPF(m0_soft, [xip, xpj], [[0],[1]])
            m_fitted = ipf_engine.iteration()
        return pd.DataFrame(m_fitted, index=H_vals, columns=B_vals)

    # -----------------------
    # Build processing units
    # -----------------------
    def build_processing_units(area_map: dict[str, str]) -> list[dict]:
        """
        Returns a list of 'units' to process.
        Each unit describes one output (either a merged LA or a single EPC dir bucket).
          {
            "label": <output label>,
            "epc_dirs": [<one or many EPC dirs>],
            "ons_area_names": [<one or many Area Names>],
            "is_merged": True/False
          }
        Rules:
          - First, create units for the MERGED_LA_GROUPS with all member areas that
            exist in area_map (collect their EPC dirs).
          - Then, for any EPC dir not already covered by a merged unit, create a unit
            where ons_area_names = all Area Names mapped to that EPC dir, epc_dirs = [that dir].
        """
        # Invert area_map: epc_dir -> set of area names
        epc_to_areas = defaultdict(list)
        for area, epc_dir in area_map.items():
            epc_to_areas[epc_dir].append(area)

        used_epc_dirs = set()
        units = []

        # 1) Add merged units
        for merged_label, members in MERGED_LA_GROUPS.items():
            # Keep only members that appear in area_map
            members_present = [m for m in members if m in area_map]
            if not members_present:
                # Nothing to merge for this label (none of the member names in CSV)
                continue
            # Collect EPC dirs from those members
            epc_dirs = sorted(set(area_map[m] for m in members_present))
            if not epc_dirs:
                continue
            units.append({
                "label": merged_label,
                "epc_dirs": epc_dirs,
                "ons_area_names": members_present,   # sum ONS for these members
                "is_merged": True
            })
            used_epc_dirs.update(epc_dirs)

        # 2) Add remaining single-epc-dir units (ONS summed across all its mapped Area Names)
        for epc_dir, areas in epc_to_areas.items():
            if epc_dir in used_epc_dirs:
                continue
            units.append({
                "label": epc_dir,                 # we’ll sanitize for folder name
                "epc_dirs": [epc_dir],
                "ons_area_names": sorted(areas),  # sum all ONS rows that map to this dir
                "is_merged": False
            })

        return units

    # -----------------------
    # Main
    # -----------------------
    def main():
        with open(AREA_JSON, "r", encoding="utf-8") as f:
            area_map = json.load(f)   # {Area Name -> EPC dir}

        os.makedirs(OUTPUT_ROOT, exist_ok=True)

        units = build_processing_units(area_map)

        for unit in tqdm(units, desc="Processing units"):
            label = unit["label"]
            epc_dirs = unit["epc_dirs"]
            ons_areas = unit["ons_area_names"]
            is_merged = unit["is_merged"]

            # ---- Load EPC: concat across all epc_dirs in the unit ----
            epc_frames = []
            missing_epc = []
            for ed in epc_dirs:
                epc_file = os.path.join(EPC_ROOT, ed, "certificates.csv")
                if not os.path.isfile(epc_file):
                    missing_epc.append((ed, epc_file))
                    continue
                try:
                    epc_frames.append(load_epc_certificates(epc_file))
                except Exception as e:
                    print(f"SKIP: failed to load EPC for {ed}: {e}")

            if not epc_frames:
                if missing_epc:
                    for ed, p in missing_epc:
                        print(f"SKIP: EPC file not found for dir {ed} at {p}")
                # Nothing to process for this unit
                continue

            epc = pd.concat(epc_frames, ignore_index=True)

            # ---- Sum ONS bedroom counts across all areas in this unit ----
            ons_summed_by_ptype = {}
            for ptype, ons_path in ONS_FILES.items():
                s = sum_ons_for_areas(ons_path, ons_areas)
                ons_summed_by_ptype[ptype] = s

            # ---- Output directory ----
            out_dir_label = label if is_merged else sanitize(label)
            la_out_dir = os.path.join(OUTPUT_ROOT, sanitize(out_dir_label))
            os.makedirs(la_out_dir, exist_ok=True)

            # ---- For each property type, compute IPF ----
            for ptype, col_counts in ons_summed_by_ptype.items():
                p_df = epc[epc["ONS_PTYPE"] == ptype]
                if p_df.empty:
                    continue

                # EPC row marginals by habitable rooms
                row_counts = p_df.groupby("H").size().sort_index()
                if row_counts.sum() == 0:
                    continue

                # Scale ONS to EPC total
                total_rows = float(row_counts.sum())
                total_cols = float(col_counts.sum())
                if total_cols <= 0:
                    continue
                col_counts_scaled = col_counts * (total_rows / total_cols)

                # IPF with H<B structural zeros
                mat = run_ipf_with_structural_zeros(row_counts, col_counts_scaled)

                # Save
                fname = f"{sanitize(ptype)}_contingency.csv"
                out_file = os.path.join(la_out_dir, fname)
                os.makedirs(os.path.dirname(out_file), exist_ok=True)
                mat.to_csv(out_file, float_format="%.6f")

        print("All done.")

    if __name__ == "__main__":
        main()

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
