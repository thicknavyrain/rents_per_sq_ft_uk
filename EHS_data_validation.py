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
    EHS_physical_df
    return (EHS_physical_df,)


@app.cell
def _(EHS_interview_df, EHS_physical_df):
    overlapping = set(EHS_interview_df['serialanon'].unique()) & set(EHS_physical_df['serialanon'].unique())
    overlapping
    return (overlapping,)


@app.cell
def _(overlapping):
    print(len(overlapping))
    return


@app.cell
def _(pd):
    EHS_general_df = pd.read_spss('EHS_stock_data/UKDA-9314-spss/spss/spss28/general21_plus_22_eul.sav')
    EHS_general_df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
