import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    # compute_rental_floor_stats_by_bedrooms_updated.py
    #
    # Computes private-rental floor space stats (mean, std) by bedroom count for each LA.
    # It uses the new 6+ category contingency tables as input but aggregates them to
    # produce the final 4+ bedroom output format.
    #
    # Output:
    #   rental_floor_stats_by_bedrooms.csv  # one row per LA

    import os
    import re
    import json
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    from collections import defaultdict

    # -----------------------
    # Config / file locations
    # -----------------------
    AREA_JSON = "linking_files/area_to_epc_dir.json" # <-- UPDATED
    EPC_ROOT = "EPC_data"
    IPF_ROOT = "ipf_outputs"

    # Define the INPUT and OUTPUT bedroom columns separately
    IPF_BEDROOM_COLS = ["1", "2", "3", "4", "5", "6+"] # How they appear in the new contingency tables
    OUTPUT_BEDROOM_BANDS = ["1", "2", "3", "4+"]      # The desired columns for the final output
    PTYPES = ["Bungalow", "Flat/Maisonette", "House : Detached", "House : Semi-Detached", "House : Terraced"]

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
        """Maps EPC property and built form to a single ONS category."""
        pt = (property_type or "").strip()
        bf = (built_form or "").strip()
        if pt in {"Flat", "Maisonette"}: return "Flat/Maisonette"
        if pt == "Bungalow": return "Bungalow"
        if pt == "House":
            if bf == "Detached": return "House : Detached"
            if bf == "Semi-Detached": return "House : Semi-Detached"
            if bf in {"Mid-Terrace", "End-Terrace", "Enclosed Mid-Terrace", "Enclosed End-Terrace"}: return "House : Terraced"
        return "Other/Unknown"

    def nearest_available_H(h: int, available: np.ndarray) -> int:
        """Finds the closest Habitable Room count in the contingency table index if an exact match isn't found."""
        idx = np.searchsorted(available, h)
        if idx == 0: return int(available[0])
        if idx >= len(available): return int(available[-1])
        prev_h, next_h = available[idx-1], available[idx]
        return int(prev_h if abs(h - prev_h) <= abs(next_h - h) else next_h)

    def weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
        """Calculates the weighted mean and standard deviation."""
        if values.size == 0 or np.sum(weights) <= 0:
            return (np.nan, np.nan)
        m = np.average(values, weights=weights)
        v = np.average((values - m) ** 2, weights=weights)
        return float(m), float(np.sqrt(max(v, 0.0)))

    def load_epc_certificates(epc_file: str) -> pd.DataFrame:
        """Loads and filters private rental certificates from an EPC file, handling missing data."""
        usecols = ["PROPERTY_TYPE", "BUILT_FORM", "NUMBER_HABITABLE_ROOMS", "TOTAL_FLOOR_AREA", "TENURE"]
        df = pd.read_csv(epc_file, usecols=usecols, low_memory=False)

        # Robustly handle missing text data to prevent errors
        df['PROPERTY_TYPE'] = df['PROPERTY_TYPE'].fillna('').astype(str)
        df['BUILT_FORM'] = df['BUILT_FORM'].fillna('').astype(str)
        df['TENURE'] = df['TENURE'].fillna('').astype(str)

        # Coerce numerics and filter for valid private rental properties
        df["NUMBER_HABITABLE_ROOMS"] = pd.to_numeric(df["NUMBER_HABITABLE_ROOMS"], errors="coerce")
        df["TOTAL_FLOOR_AREA"] = pd.to_numeric(df["TOTAL_FLOOR_AREA"], errors="coerce")
        is_prs = df["TENURE"].str.contains(r"rented\s*\(private\)|rental\s*\(private\)", na=False, case=False)
        df = df[is_prs].copy()
        df = df.dropna(subset=["NUMBER_HABITABLE_ROOMS", "TOTAL_FLOOR_AREA"])
        df = df[df["TOTAL_FLOOR_AREA"] > 0]

        # Assign ONS category and clean up Habitable Rooms column
        df["H"] = df["NUMBER_HABITABLE_ROOMS"].astype(int).clip(lower=1)
        df["ONS_PTYPE"] = df.apply(lambda r: map_epc_to_ons_category(r["PROPERTY_TYPE"], r["BUILT_FORM"]), axis=1)
        df = df[df["ONS_PTYPE"].isin(PTYPES)]
        return df

    def build_processing_units(area_map: dict[str, str]) -> list[dict]:
        """
        SIMPLIFIED: Creates a processing unit for each EPC directory.
        """
        epc_to_areas = defaultdict(list)
        for area, epc_dir in area_map.items():
            epc_to_areas[epc_dir].append(area)

        units = []
        for epc_dir in sorted(epc_to_areas.keys()):
            units.append({
                "label": epc_dir,
                "epc_dirs": [epc_dir],
                "ipf_dir": sanitize(epc_dir),
            })
        return units

    # -----------------------
    # Main Execution
    # -----------------------
    def main():
        with open(AREA_JSON, "r", encoding="utf-8") as f:
            area_map = json.load(f)

        # Prepare final result structure
        cols = ["Area Name"]
        for b in OUTPUT_BEDROOM_BANDS:
            label = "4plus" if b == "4+" else b
            cols += [f"mean_{label}", f"std_{label}"]
        rows = []

        units = build_processing_units(area_map)

        for unit in tqdm(units, desc="Processing LAs"):
            label = unit["label"]
            ipf_dirname = unit["ipf_dir"]

            # --- Load and concat all EPC data for this unit ---
            epc_frames = [
                load_epc_certificates(os.path.join(EPC_ROOT, ed, "certificates.csv"))
                for ed in unit["epc_dirs"]
                if os.path.isfile(os.path.join(EPC_ROOT, ed, "certificates.csv"))
            ]

            if not epc_frames:
                print(f"SKIP: No valid EPC data found for {label}")
                continue
            df_prs = pd.concat(epc_frames, ignore_index=True)

            if df_prs.empty:
                continue

            # --- Load and aggregate contingency tables for this unit ---
            cont_tables = {}
            for ptype in PTYPES:
                path = os.path.join(IPF_ROOT, ipf_dirname, f"{sanitize(ptype)}_contingency.csv")
                if os.path.isfile(path):
                    mat = pd.read_csv(path, index_col=0)
                    # Aggregate 4, 5, and 6+ columns into a single '4+' column
                    mat['4+'] = mat[['4', '5', '6+']].sum(axis=1)
                    mat = mat.drop(columns=['4', '5', '6+'])
                    cont_tables[ptype] = mat

            if not cont_tables:
                print(f"SKIP: No contingency tables found for unit '{label}' in {IPF_ROOT}/{ipf_dirname}")
                continue

            # --- Calculate weighted stats ---
            values_by_bed = {b: [] for b in OUTPUT_BEDROOM_BANDS}
            weights_by_bed = {b: [] for b in OUTPUT_BEDROOM_BANDS}

            for ptype, grp in df_prs.groupby("ONS_PTYPE"):
                mat = cont_tables.get(ptype)
                if mat is None or mat.empty: continue

                row_sums = mat.sum(axis=1)
                available_H = np.array(sorted(mat.index.unique()))

                for _, r in grp.iterrows():
                    h = int(r["H"])
                    y = float(r["TOTAL_FLOOR_AREA"])
                    h_use = h if h in mat.index else nearest_available_H(h, available_H)

                    if h_use not in row_sums.index or row_sums.loc[h_use] <= 0: continue

                    probs = mat.loc[h_use, OUTPUT_BEDROOM_BANDS].values.astype(float)
                    probs = np.nan_to_num(probs, nan=0.0) / row_sums.loc[h_use]

                    for b, p in zip(OUTPUT_BEDROOM_BANDS, probs):
                        if p > 0:
                            values_by_bed[b].append(y)
                            weights_by_bed[b].append(p)

            # --- Store results for this LA ---
            out_row = {"Area Name": label}
            for b in OUTPUT_BEDROOM_BANDS:
                lab = "4plus" if b == "4+" else b
                vals = np.array(values_by_bed[b], dtype=float)
                wts  = np.array(weights_by_bed[b], dtype=float)
                mean_b, std_b = weighted_mean_std(vals, wts)
                out_row[f"mean_{lab}"] = mean_b
                out_row[f"std_{lab}"]  = std_b
            rows.append(out_row)

        # --- Save final output file ---
        pd.DataFrame(rows, columns=cols).to_csv("outputs/rental_floor_stats_by_bedrooms.csv", index=False)
        print("\n✅ All done. Saved to rental_floor_stats_by_bedrooms.csv")

    if __name__ == "__main__":
        main()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
