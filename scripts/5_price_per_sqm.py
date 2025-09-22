import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    # pipr_cost_per_m2_from_floor_areas.py
    #
    # Match PIPR areas to rental_floor_stats areas via fuzzywuzzy (score >= 90),
    # then compute cost per m² by bedroom (1,2,3,4+) and an overall cost per m².
    #
    # Requires:
    #   pip install fuzzywuzzy python-Levenshtein
    #
    # Inputs:
    #   rental_floor_stats_by_bedrooms.csv
    #   PIPR_data.csv
    #
    # Output:
    #   pipr_cost_per_m2.csv

    import pandas as pd
    import numpy as np
    from fuzzywuzzy import process

    RENTAL_CSV = "outputs/rental_floor_stats_by_bedrooms.csv"
    PIPR_CSV  = "PIPR_data/PIPR_data.csv"
    OUT_CSV   = "outputs/pipr_cost_per_m2.csv"

    def to_num(s):
        return pd.to_numeric(s, errors="coerce")

    def safe_div(numer, denom):
        numer = to_num(numer)
        denom = to_num(denom)
        out = numer / denom
        out[(denom <= 0) | denom.isna()] = np.nan
        return out

    def main():
        rental = pd.read_csv(RENTAL_CSV)
        pipr  = pd.read_csv(PIPR_CSV)

        rental.columns = [c.strip() for c in rental.columns]
        pipr.columns  = [c.strip() for c in pipr.columns]

        rental_areas = sorted(set(rental["Area Name"].astype(str).str.strip()))
        pipr_areas  = sorted(set(pipr["Area name"].astype(str).str.strip()))

        # Fuzzy match PIPR Local Authority names -> rental properties dataset names (score >= 90)
        mapping = []
        for nm in pipr_areas:
            best = process.extractOne(nm, rental_areas)
            if best:
                match_name, score = best[0], best[1]
                if score >= 80:
                    mapping.append({"Area name": nm, "Area Name (rental)": match_name, "score": score})
                else:
                    print(nm)
                    print(match_name)
        map_df = pd.DataFrame(mapping)

        # Merge mapped names back to full rows
        pipr_m = pipr.merge(map_df, on="Area name", how="inner")
        rental_m = rental.rename(columns={"Area Name": "Area Name (rental)"})
        df = pipr_m.merge(rental_m, on="Area Name (rental)", how="left")

        # Coerce numeric rent columns
        df["Rental price one bed"]           = to_num(df.get("Rental price one bed"))
        df["Rental price two bed"]           = to_num(df.get("Rental price two bed"))
        df["Rental price three bed"]         = to_num(df.get("Rental price three bed"))
        df["Rental price four or more bed"]  = to_num(df.get("Rental price four or more bed"))
        if "Rental price" in df.columns:
            df["Rental price"] = to_num(df["Rental price"])

        # Coerce numeric area means
        df["mean_1"]      = to_num(df.get("mean_1"))
        df["mean_2"]      = to_num(df.get("mean_2"))
        df["mean_3"]      = to_num(df.get("mean_3"))
        df["mean_4plus"]  = to_num(df.get("mean_4plus"))

        # Cost per m² by bedroom
        df["cpm2_1"]     = safe_div(df["Rental price one bed"],          df["mean_1"])
        df["cpm2_2"]     = safe_div(df["Rental price two bed"],          df["mean_2"])
        df["cpm2_3"]     = safe_div(df["Rental price three bed"],        df["mean_3"])
        df["cpm2_4plus"] = safe_div(df["Rental price four or more bed"], df["mean_4plus"])

        # Overall cost per m² (simple average across available bands)
        cpm2_cols = ["cpm2_1", "cpm2_2", "cpm2_3", "cpm2_4plus"]
        df["cpm2_overall"] = df[cpm2_cols].mean(axis=1, skipna=True)

        # Keep mapping & results (now including Area code)
        out_cols = [
            "Area code",           # <-- added
            "Area name",
            "cpm2_1", "cpm2_2", "cpm2_3", "cpm2_4plus", "cpm2_overall"
        ]
        out = df[out_cols].sort_values(["Area name"]).reset_index(drop=True)

        out.to_csv(OUT_CSV, index=False)
        print(f"Saved {OUT_CSV}")
        print(out.head(10))

    if __name__ == "__main__":
        main()

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
