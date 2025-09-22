import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    # ipf_contingency_tables_all_LAs_updated.py
    #
    # Build HabitableRooms x Bedrooms IPF contingency tables for EVERY EPC directory
    # referenced in linking_files/area_to_epc_dir.json.
    # ONS data is summed across all Area Names mapped to a given directory.
    #
    # Outputs:
    #   ipf_outputs/<sanitized_epc_directory_name>/<ptype>_contingency.csv

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
    AREA_JSON = "linking_files/area_to_epc_dir.json"
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

    # Define the new bedroom columns
    BEDROOM_COLS = ["1", "2", "3", "4", "5", "6+"]

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
        """Loads EPC data"""
        usecols = [
            "PROPERTY_TYPE", "BUILT_FORM", "LOCAL_AUTHORITY",
            "NUMBER_HABITABLE_ROOMS", "TOTAL_FLOOR_AREA",
        ]
        df = pd.read_csv(epc_file, usecols=usecols, low_memory=False)

        # Convert key text columns to string type and fill any missing values (NaN)
        df['PROPERTY_TYPE'] = df['PROPERTY_TYPE'].fillna('').astype(str)
        df['BUILT_FORM'] = df['BUILT_FORM'].fillna('').astype(str)

        df["ONS_PTYPE"] = df.apply(lambda r: map_epc_to_ons_category(r["PROPERTY_TYPE"], r["BUILT_FORM"]), axis=1)
        df = df.dropna(subset=["NUMBER_HABITABLE_ROOMS"])
        df["H"] = pd.to_numeric(df["NUMBER_HABITABLE_ROOMS"], errors="coerce").astype(int).clip(lower=1)
        ons_cats = set(ONS_FILES.keys())
        df = df[df["ONS_PTYPE"].isin(ons_cats)]
        return df

    def load_ons_bedroom_counts(ons_file: str, area_name: str) -> pd.Series:
        """Loads ONS data for a single area, adapted for new bedroom columns."""
        df = pd.read_csv(ons_file)
        # In the new files, the area name column is 'area_name' not 'Area Name'
        area_col = 'area_name' if 'area_name' in df.columns else 'Area Name'
        df.columns = [c.strip() for c in df.columns]
        for col in BEDROOM_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        row = df.loc[df[area_col].str.strip().str.casefold() == (area_name or "").strip().casefold()]
        if row.empty:
            raise KeyError(f"Area Name '{area_name}' not found in {ons_file}")

        s = row.iloc[0][BEDROOM_COLS].astype(float)
        s.index = BEDROOM_COLS
        return s

    def sum_ons_for_areas(ons_file: str, area_names: list[str]) -> pd.Series:
        """Sum ONS bedroom counts for a list of Area Names."""
        total = pd.Series([0.0] * len(BEDROOM_COLS), index=BEDROOM_COLS)
        missing = []
        for nm in area_names:
            try:
                s = load_ons_bedroom_counts(ons_file, nm)
                total = total.add(s, fill_value=0.0)
            except KeyError:
                missing.append(nm)
        if missing:
            print(f"WARNING: Missing ONS rows in {os.path.basename(ons_file)} for: {', '.join(missing)}")
        return total

    def run_ipf_with_structural_zeros(row_counts: pd.Series, col_counts_scaled: pd.Series) -> pd.DataFrame:
        """Runs IPF, adapted for new bedroom columns."""
        H_vals = row_counts.index.tolist()
        m0 = np.ones((len(H_vals), len(BEDROOM_COLS)), dtype=float)

        # Create a mask where Habitable Rooms (H) is equal to or less than Bedrooms (B)
        H_numeric = np.array(H_vals, dtype=int)
        B_numeric = np.array([1, 2, 3, 4, 5, 6], dtype=int)  # Treat '6+' as 6 for this comparison
        mask = (H_numeric[:, None] < B_numeric[None, :])
        m0[mask] = 0.0  # Set initial values to 0 (structural zeros)

        xip = row_counts.values.astype(float)
        xpj = col_counts_scaled.reindex(BEDROOM_COLS).values.astype(float)

        try:
            ipf_engine = IPF(m0.copy(), [xip, xpj], [[0],[1]])
            m_fitted = ipf_engine.iteration()
        except Exception: # If perfect structural zeros fail, soften them slightly
            m0_soft = m0.copy()
            m0_soft[mask] = 1e-12
            ipf_engine = IPF(m0_soft, [xip, xpj], [[0],[1]])
            m_fitted = ipf_engine.iteration()

        return pd.DataFrame(m_fitted, index=H_vals, columns=BEDROOM_COLS)

    def build_processing_units(area_map: dict[str, str]) -> list[dict]:
        """
        Returns a list of 'units' to process. Each unit corresponds to one unique
        EPC directory and includes all Area Names that map to it.
        """
        epc_to_areas = defaultdict(list)
        for area, epc_dir in area_map.items():
            epc_to_areas[epc_dir].append(area)

        units = []
        for epc_dir, areas in epc_to_areas.items():
            units.append({
                "label": epc_dir,  # Use the EPC directory name as the label
                "epc_dirs": [epc_dir],
                "ons_area_names": sorted(areas),
            })
        return units

    def main():
        with open(AREA_JSON, "r", encoding="utf-8") as f:
            area_map = json.load(f)

        os.makedirs(OUTPUT_ROOT, exist_ok=True)

        units = build_processing_units(area_map)

        for unit in tqdm(units, desc="Processing LAs"):
            label = unit["label"]
            epc_dirs = unit["epc_dirs"]
            ons_areas = unit["ons_area_names"]

            epc_frames = []
            for ed in epc_dirs:
                epc_file = os.path.join(EPC_ROOT, ed, "certificates.csv")
                if os.path.isfile(epc_file):
                    try:
                        epc_frames.append(load_epc_certificates(epc_file))
                    except Exception as e:
                        print(f"SKIP: Failed to load EPC for {ed}: {e}")
                else:
                    print(f"SKIP: EPC file not found for dir {ed} at {epc_file}")

            if not epc_frames:
                continue
            epc = pd.concat(epc_frames, ignore_index=True)

            ons_summed_by_ptype = {
                ptype: sum_ons_for_areas(ons_path, ons_areas)
                for ptype, ons_path in ONS_FILES.items()
            }

            la_out_dir = os.path.join(OUTPUT_ROOT, sanitize(label))
            os.makedirs(la_out_dir, exist_ok=True)

            for ptype, col_counts in ons_summed_by_ptype.items():
                p_df = epc[epc["ONS_PTYPE"] == ptype]
                if p_df.empty or col_counts.sum() <= 0:
                    continue

                row_counts = p_df.groupby("H").size().sort_index()
                if row_counts.sum() == 0:
                    continue

                total_rows = float(row_counts.sum())
                total_cols = float(col_counts.sum())
                col_counts_scaled = col_counts * (total_rows / total_cols)

                mat = run_ipf_with_structural_zeros(row_counts, col_counts_scaled)

                fname = f"{sanitize(ptype)}_contingency.csv"
                out_file = os.path.join(la_out_dir, fname)
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
