import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell
def _(pd):
    EHS_df = pd.read_csv('./EHS_bedrooms_floor.csv')
    EHS_df
    return (EHS_df,)


@app.cell
def _(EHS_df):
    # Define grouping columns
    group_cols = [
        "nbedsx", "tenex", "tenure8x", "tenure4x",
        "accomhh1", "dwtypenx", "dwtype8x", "dwtype3x",
        "EPceeb12e", "EPceib12e", "gorehs"
    ]

    # Function to calculate summary stats
    def summarise_floor(data, floor_col):
        return (
            data.groupby(group_cols)[floor_col]
            .agg(
                mean="mean",
                std="std",
                p25=lambda x: x.quantile(0.25),
                median="median",
                p75=lambda x: x.quantile(0.75)
            )
            .reset_index()
        )

    # Generate the two summary tables
    floorx_summary = summarise_floor(EHS_df, "floorx")
    floory_summary = summarise_floor(EHS_df, "floory")

    # Save to CSV (optional)
    floorx_summary.to_csv("floorx_summary.csv", index=False)
    floory_summary.to_csv("floory_summary.csv", index=False)

    # Display previews
    print("Floorx summary (first 5 rows):")
    print(floorx_summary.head(), "\n")
    print("Floory summary (first 5 rows):")
    print(floory_summary.head())

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
