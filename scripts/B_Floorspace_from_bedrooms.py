import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell
def _(pd):
    EHS_df = pd.read_csv('EHS_analysis_outputs/EHS_bedrooms_floor.csv')
    EHS_df = EHS_df[EHS_df['tenex'] == 'privately rent']
    EHS_df
    return (EHS_df,)


@app.cell
def _(EHS_df):
    # Define grouping columns
    # group_cols = [
    #     "nbedsx", "tenex", "tenure8x", "tenure4x",
    #     "accomhh1", "dwtypenx", "dwtype8x", "dwtype3x",
    #     "EPceeb12e", "EPceib12e", "gorehs"
    # ]



    ## Just bedroom and region
    group_cols = [
        "nbedsx", "gorehs"
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

    # # Save to CSV (optional)
    # floorx_summary.to_csv("floorx_summary.csv", index=False)
    # floory_summary.to_csv("floory_summary.csv", index=False)

    # Display previews
    print("Floorx summary (first 5 rows):")
    print(floorx_summary.head(), "\n")
    print("Floory summary (first 5 rows):")
    print(floory_summary.head())

    return (floorx_summary,)


@app.cell
def _(floorx_summary):
    floorx_summary
    return


@app.cell
def _(floorx_summary, pd):
    import matplotlib.pyplot as plt

    # Some values in nbedsx are already '5 or more' (string), so handle mixed types safely
    plot_df = floorx_summary.copy()

    # Convert numeric-like values to int, keep "5 or more" as string
    def clean_beds(x):
        try:
            return str(int(float(x)))
        except:
            return "5 or more"

    plot_df["nbedsx"] = plot_df["nbedsx"].apply(clean_beds)

    # Set order for categorical axis
    order = ["1", "2", "3", "4", "5 or more"]
    plot_df["nbedsx"] = pd.Categorical(plot_df["nbedsx"], categories=order, ordered=True)

    # Plot
    plt.figure(figsize=(12, 7))

    for region, group in plot_df.groupby("gorehs"):
        plt.errorbar(
            group["nbedsx"],
            group["mean"],
            yerr=group["std"],
            fmt="o",
            capsize=4,
            label=region
        )

    plt.xlabel("Number of Bedrooms")
    plt.ylabel("Mean Floor Space (m²)")
    plt.title("Mean Floor Space by Bedrooms and Region (with Std Error Bars)")
    plt.legend(title="Region of England", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig('EHS_analysis_outputs/EHS_foorspace_by_bedroom.png', bbox_inches='tight', dpi=300)
    plt.show()

    return clean_beds, order, plot_df, plt


@app.cell
def _(clean_beds, order, pd, plot_df, plt):
    plot_df["nbedsx"] = plot_df["nbedsx"].apply(clean_beds)
    plot_df["nbedsx"] = pd.Categorical(plot_df["nbedsx"], categories=order, ordered=True)

    # Plot with median and percentile-based error bars
    plt.figure(figsize=(12, 7))

    for region2, group2 in plot_df.groupby("gorehs"):
        lower_err = group2["median"] - group2["p25"]
        upper_err = group2["p75"] - group2["median"]
        asym_err = [lower_err, upper_err]

        plt.errorbar(
            group2["nbedsx"],
            group2["median"],
            yerr=asym_err,
            fmt="o",
            capsize=4,
            label=region2
        )

    plt.xlabel("Number of Bedrooms")
    plt.ylabel("Median Floor Space (m²)")
    plt.title("Median Floor Space by Bedrooms and Region (with IQR Error Bars)")
    plt.legend(title="Region of England", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig('EHS_analysis_outputs/EHS_median_foorspace_by_bedroom.png', bbox_inches='tight', dpi=300)
    plt.show()

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
