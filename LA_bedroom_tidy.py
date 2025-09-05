import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import os
    return os, pd


@app.cell
def _(os, pd):
    directory = './LA_bedroom_properties/raw/'

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):  
            test_df = pd.read_csv(file_path, encoding="latin1")
            processed_df = test_df[test_df['Band'] == 'All'].drop(columns=['Band','Not Known6','All'])
            processed_df.to_csv('./LA_bedroom_properties/'+filename, index=False)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
