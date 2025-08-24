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
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
