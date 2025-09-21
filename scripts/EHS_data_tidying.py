import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell
def _(pd):
    EHS_interview_df = pd.read_spss('EHS_stock_data/UKDA-9314-spss/spss/spss28/interview_21_plus_22_eul.sav')
    EHS_interview_df = EHS_interview_df[['serialanon', 'nbedsx', 'tenex', 'accomhh1']]
    return (EHS_interview_df,)


@app.cell
def _(EHS_interview_df):
    EHS_interview_df
    return


@app.cell
def _():
    # EHS_renters = EHS_interview_df[EHS_interview_df['tenex'] == 'privately rent']
    # EHS_renters
    return


@app.cell
def _(pd):
    EHS_physical_df = pd.read_spss('EHS_stock_data/UKDA-9314-spss/spss/spss28/physical_21_plus_22_eul.sav')
    EHS_physical_df = EHS_physical_df[['serialanon','dwtypenx', 'dwtype8x', 'dwtype3x', 'floorx', 'floory', 'EPceeb12e', 'EPceib12e']]
    EHS_physical_df
    return (EHS_physical_df,)


@app.cell
def _(EHS_physical_df):
    EHS_physical_df['dwtype8x'].unique()
    return


@app.cell
def _(pd):
    EHS_general_df = pd.read_spss('EHS_stock_data/UKDA-9314-spss/spss/spss28/general21_plus_22_eul.sav')
    EHS_general_df = EHS_general_df[['serialanon','gorehs', 'tenure8x', 'tenure4x']]
    EHS_general_df
    return (EHS_general_df,)


@app.cell
def _(EHS_general_df, EHS_interview_df, EHS_physical_df, pd):
    import numpy as np

    merged_df = (
        EHS_interview_df
        .merge(EHS_physical_df, on="serialanon", how="inner", suffixes=("_interview", "_physical"))
        .merge(EHS_general_df,  on="serialanon", how="inner", suffixes=("", "_general"))
    )

    def clean_to_float(s: pd.Series) -> pd.Series:
        # Ensure string dtype (handles categoricals/objects)
        s = s.astype("string")
        # Normalize common textual patterns before stripping
        # e.g., "500 or more" -> "500", "c. 123" -> "123", "1,234" -> "1234"
        s = (
            s.str.replace(",", "", regex=False)                # remove thousands separators
             .str.replace(r"\bor more\b", "", regex=True)      # drop trailing phrases
             .str.replace(r"\bapprox(?:\.|)\b", "", regex=True)
             .str.replace(r"\bc(?:irca|\.?)\b", "", regex=True)
             .str.strip()
        )
        # Keep only the first numeric token (handles "250-300", "500 +", etc.)
        first_num = s.str.extract(r"([-+]?\d*\.?\d+)")[0]
        # Convert to float; non-parsable -> NaN
        out = pd.to_numeric(first_num, errors="coerce")
        return out

    for col in ["floorx", "floory"]:
        if col in merged_df.columns:
            merged_df[col] = clean_to_float(merged_df[col])
        else:
            raise KeyError(f"Expected column '{col}' not found in merged_df")

    # Optional: if you prefer to fail fast when any non-numeric remains, uncomment:
    # if merged_df[["floorx", "floory"]].isna().any().any():
    #     bad_rows = merged_df[merged_df[["floorx", "floory"]].isna().any(axis=1)][["serialanon", "floorx", "floory"]]
    #     raise ValueError(f"Non-numeric values found in floor columns:\n{bad_rows.head()}")

    # Save with 2dp formatting
    merged_df.to_csv("EHS_bedrooms_floor.csv", index=False, float_format="%.2f")

    return (merged_df,)


@app.cell
def _(merged_df):
    merged_df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
