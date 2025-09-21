import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import re
    import pandas as pd
    import json

    # --------- USER SETTINGS ----------
    EPC_PARENT_DIR = "EPC_data"
    AREA_CSV = "LA_bedroom_properties/flat_maisonette_bedrooms.csv"
    OUT_JSON = "linking_files/area_to_epc_dir.json"

    # Manual overrides for historical authorities that have merged.
    # This maps the old area name directly to the new EPC directory name.
    AREA_NAME_OVERRIDES = {
        # Bournemouth, Christchurch, Poole -> domestic-E06000058-Bournemouth-Christchurch-and-Poole
        "Bournemouth UA": "domestic-E06000058-Bournemouth-Christchurch-and-Poole",
        "Christchurch": "domestic-E06000058-Bournemouth-Christchurch-and-Poole",
        "Poole UA": "domestic-E06000058-Bournemouth-Christchurch-and-Poole",
        # Dorset unitary -> domestic-E06000059-Dorset
        "East Dorset": "domestic-E06000059-Dorset",
        "North Dorset": "domestic-E06000059-DSorset",
        "Purbeck": "domestic-E06000059-Dorset",
        "West Dorset": "domestic-E06000059-Dorset",
        "Weymouth and Portland": "domestic-E06000059-Dorset",
    }
    # ----------------------------------

    def create_ecode_to_dir_map(parent_dir: str) -> dict:
        """Scans a directory and maps extracted ecodes to their full directory names."""
        if not os.path.isdir(parent_dir):
            print(f"❌ Error: EPC parent directory not found at '{parent_dir}'")
            return {}
        ecode_map = {}
        pattern = re.compile(r"^domestic-([A-Z0-9]+)-", re.IGNORECASE)
        for dirname in os.listdir(parent_dir):
            if os.path.isdir(os.path.join(parent_dir, dirname)):
                match = pattern.match(dirname)
                if match:
                    ecode = match.group(1)
                    ecode_map[ecode] = dirname
        return ecode_map

    def main():
        """Main function to perform the matching and generate outputs."""
        ecode_to_dir = create_ecode_to_dir_map(EPC_PARENT_DIR)
        if not ecode_to_dir:
            print("No EPC directories found or directory is invalid. Exiting.")
            return

        try:
            df = pd.read_csv(AREA_CSV)
        except FileNotFoundError:
            print(f"❌ Error: Input CSV not found at '{AREA_CSV}'.")
            return

        final_mapping = {}
        unmatched_areas = {}

        # --- UPDATED MATCHING LOGIC ---
        for _, row in df.iterrows():
            area_name = row['area_name']
            ecode = row['ecode']
        
            # 1. First, try to match by the unique ecode
            if ecode in ecode_to_dir:
                final_mapping[area_name] = ecode_to_dir[ecode]
            # 2. If that fails, check for a manual override by area name
            elif area_name in AREA_NAME_OVERRIDES:
                final_mapping[area_name] = AREA_NAME_OVERRIDES[area_name]
            # 3. If both fail, it's a true mismatch
            else:
                unmatched_areas[area_name] = ecode
    
        with open(OUT_JSON, "w", encoding="utf-8") as jf:
            json.dump(final_mapping, jf, ensure_ascii=False, indent=2)

        print(f"✅ Success! Created mapping for {len(final_mapping)} areas.")
        print(f"--> Saved to {OUT_JSON}\n")

        # --- MISMATCH REPORT ---
        print("--- Mismatch Report ---")
        if unmatched_areas:
            print(f"\n⚠️ Found {len(unmatched_areas)} areas in the CSV with no matching EPC directory:")
            for name, code in unmatched_areas.items():
                print(f"  - {name} (ecode: {code})")
        else:
            print("✅ All areas in the CSV were successfully matched to an EPC directory.")

        # Find EPC directories that were not used in any mapping
        used_dirs = set(final_mapping.values())
        all_epc_dirs = set(ecode_to_dir.values())
        unmatched_epc_dirs = all_epc_dirs - used_dirs

        if unmatched_epc_dirs:
            print(f"\n⚠️ Found {len(unmatched_epc_dirs)} EPC directories with no matching area in the CSV:")
            for dirname in sorted(list(unmatched_epc_dirs)):
                print(f"  - {dirname}")
        else:
            print("✅ All EPC directories were successfully matched to an area in the CSV.")

    if __name__ == "__main__":
        main()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
