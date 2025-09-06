import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    # compute_rental_floor_stats_by_bedrooms.py
    #
    # Compute LA-level private-rental floor space stats by bedroom band (1,2,3,4+)
    # using EPC microdata + precomputed IPF contingency tables.
    #
    # Inputs:
    #   area_to_epc_dir.json               # {"Camden": "domestic-E09000007-Camden", ...}
    #   EPC_data/<EPC_DIR>/certificates.csv
    #   ipf_outputs/<EPC_DIR_sanitized>/*_contingency.csv   # produced by previous script
    #
    # Output:
    #   rental_floor_stats_by_bedrooms.csv  # one row per LA, mean/std for 1,2,3,4+
    #
    # Run:
    #   python compute_rental_floor_stats_by_bedrooms.py
    #
    # Notes:
    # - Filters private rentals by TENURE strings containing "rented (private)" or "rental (private)", case-insensitive.
    # - Coerces TOTAL_FLOOR_AREA to float and filters to > 0 (invalid rows are dropped with a warning).

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
    IPF_ROOT = "ipf_outputs"             # ipf_outputs/<EPC_DIR_sanitized>/<ptype>_contingency.csv

    BEDROOM_BANDS = ["1", "2", "3", "4+"]  # column order
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
    for area_name, epc_dir in tqdm(AREA_MAP.items(), desc="LAs", total = 329):
        epc_file = os.path.join(EPC_ROOT, epc_dir, "certificates.csv")
        if not os.path.isfile(epc_file):
            print(f"SKIP: EPC file not found for {area_name} at {epc_file}")
            continue

        # Load EPC for this LA
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

        # Filter private rentals (case-insensitive; includes "Rented (private)" and "rental (private)")
        tenure_lc = df["TENURE"].str.casefold()
        is_prs = tenure_lc.str.contains(r"rented\s*\(private\)", na=False) | tenure_lc.str.contains(r"rental\s*\(private\)", na=False)
        df_prs = df[is_prs].copy()

        if df_prs.empty:
            print(f"INFO: No private rental rows for {area_name}; writing NaNs.")
            stats = {f"mean_{'4plus' if b=='4+' else b}": np.nan for b in BEDROOM_BANDS}
            stats.update({f"std_{'4plus' if b=='4+' else b}": np.nan for b in BEDROOM_BANDS})
            rows.append({"Area Name": area_name, **stats})
            continue

        # Keep valid H and strictly positive floor areas; warn on dropped counts
        before = len(df_prs)
        df_prs = df_prs.dropna(subset=["NUMBER_HABITABLE_ROOMS", "TOTAL_FLOOR_AREA"])
        df_prs = df_prs[df_prs["TOTAL_FLOOR_AREA"] > 0]
        dropped = before - len(df_prs)
        if dropped > 0:
            print(f"WARNING: Dropped {dropped} private-rental rows with non-positive/missing TOTAL_FLOOR_AREA in {area_name}.")

        if df_prs.empty:
            stats = {f"mean_{'4plus' if b=='4+' else b}": np.nan for b in BEDROOM_BANDS}
            stats.update({f"std_{'4plus' if b=='4+' else b}": np.nan for b in BEDROOM_BANDS})
            rows.append({"Area Name": area_name, **stats})
            continue

        df_prs["H"] = df_prs["NUMBER_HABITABLE_ROOMS"].astype(int).clip(lower=1)
        df_prs["ONS_PTYPE"] = df_prs.apply(lambda r: map_epc_to_ons_category(r["PROPERTY_TYPE"], r["BUILT_FORM"]), axis=1)
        df_prs = df_prs[df_prs["ONS_PTYPE"].isin(PTYPES)]

        if df_prs.empty:
            print(f"INFO: No private rental rows in target property types for {area_name}; writing NaNs.")
            stats = {f"mean_{'4plus' if b=='4+' else b}": np.nan for b in BEDROOM_BANDS}
            stats.update({f"std_{'4plus' if b=='4+' else b}": np.nan for b in BEDROOM_BANDS})
            rows.append({"Area Name": area_name, **stats})
            continue

        # Load contingency tables for this LA
        ipf_dir = os.path.join(IPF_ROOT, sanitize(epc_dir))
        cont_tables = {}
        for ptype in PTYPES:
            fname = f"{sanitize(ptype)}_contingency.csv"
            path = os.path.join(ipf_dir, fname)
            if os.path.isfile(path):
                mat = pd.read_csv(path, index_col=0)
                mat.columns = [c.strip() for c in mat.columns]
                # Ensure columns in the right order/complete
                mat = mat.reindex(columns=BEDROOM_BANDS)
                # Coerce to numeric
                mat = mat.apply(pd.to_numeric, errors="coerce")
                # Drop rows with all NAs
                mat = mat.dropna(how="all")
                # Ensure int index
                mat.index = pd.to_numeric(mat.index, errors="coerce").astype("Int64")
                mat = mat.dropna(axis=0, how="any")
                mat.index = mat.index.astype(int)
                cont_tables[ptype] = mat

        if not cont_tables:
            print(f"SKIP: No contingency tables found for {area_name} in {ipf_dir}")
            continue

        # We'll collect ALL floor areas and their weights per bedroom band,
        # then compute weighted mean/std with numpy.average.
        values_by_bed = {b: [] for b in BEDROOM_BANDS}
        weights_by_bed = {b: [] for b in BEDROOM_BANDS}

        for ptype, grp in df_prs.groupby("ONS_PTYPE"):
            mat = cont_tables.get(ptype)
            if mat is None or mat.empty:
                continue

            # Precompute row sums and available H for nearest mapping
            row_sums = mat[BEDROOM_BANDS].sum(axis=1)
            available_H = np.array(sorted(mat.index.unique()))

            # Vectorised loop over this group
            for _, r in grp.iterrows():
                h = int(r["H"])
                y = float(r["TOTAL_FLOOR_AREA"])

                # Map to nearest H in contingency table
                h_use = h if h in mat.index else nearest_available_H(h, available_H)

                probs = mat.loc[h_use, BEDROOM_BANDS].values.astype(float)
                denom = float(row_sums.loc[h_use]) if h_use in row_sums.index else float(np.nansum(probs))
                if not np.isfinite(denom) or denom <= 0:
                    continue
                probs = np.nan_to_num(probs, nan=0.0) / denom

                # Append value & weight per bedroom band
                for b, p in zip(BEDROOM_BANDS, probs):
                    if p <= 0:
                        continue
                    values_by_bed[b].append(y)
                    weights_by_bed[b].append(p)

        # Compute weighted mean/std per band
        out = {"Area Name": area_name}
        for b in BEDROOM_BANDS:
            label = "4plus" if b == "4+" else b
            vals = np.array(values_by_bed[b], dtype=float)
            wts = np.array(weights_by_bed[b], dtype=float)
            mean_b, std_b = weighted_mean_std(vals, wts)
            out[f"mean_{label}"] = mean_b
            out[f"std_{label}"] = std_b

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
