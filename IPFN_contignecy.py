import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    # ipf_contingency_tables_all_LAs.py
    #
    # Build HabitableRooms x Bedrooms IPF contingency tables for EVERY local authority
    # defined in area_to_epc_dir.json, using:
    #   - EPC_data/<LA_DIR>/certificates.csv  (per-LA EPC microdata)
    #   - LA_bedroom_properties/*.csv         (ONS counts by bedrooms per property type)
    #
    # Outputs:
    #   ipf_outputs/<EPC_DIR or AreaName sanitized>/<ptype>_contingency.csv
    #
    # Requires: pip install ipfn tqdm

    import os
    import re
    import json
    import pandas as pd
    import numpy as np
    from ipfn.ipfn import ipfn as IPF
    from tqdm import tqdm

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
    # Main
    # -----------------------
    def main():
        with open(AREA_JSON, "r", encoding="utf-8") as f:
            area_map = json.load(f)

        os.makedirs(OUTPUT_ROOT, exist_ok=True)

        for area_name, epc_dir in tqdm(area_map.items(), desc="Processing areas"):
            epc_file = os.path.join(EPC_ROOT, epc_dir, "certificates.csv")
            if not os.path.isfile(epc_file):
                print(f"SKIP: EPC file not found for {area_name} at {epc_file}")
                continue

            try:
                epc = load_epc_certificates(epc_file)
            except Exception as e:
                print(f"SKIP: failed to load EPC for {area_name}: {e}")
                continue

            la_out_dir = os.path.join(OUTPUT_ROOT, sanitize(epc_dir))
            os.makedirs(la_out_dir, exist_ok=True)

            for ptype, ons_path in ONS_FILES.items():
                p_df = epc[epc["ONS_PTYPE"] == ptype]
                if p_df.empty:
                    continue
                row_counts = p_df.groupby("H").size().sort_index()
                if row_counts.sum() == 0:
                    continue
                try:
                    col_counts = load_ons_bedroom_counts(ons_path, area_name)
                except Exception as e:
                    print(f"SKIP: ONS row missing for {area_name} in {os.path.basename(ons_path)}: {e}")
                    continue
                total_rows = float(row_counts.sum())
                total_cols = float(col_counts.sum())
                if total_cols <= 0:
                    continue
                col_counts_scaled = col_counts * (total_rows / total_cols)
                mat = run_ipf_with_structural_zeros(row_counts, col_counts_scaled)
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
