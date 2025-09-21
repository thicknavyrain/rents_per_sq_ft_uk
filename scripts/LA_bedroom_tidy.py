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
    excluded_areas = ['ENGLAND AND WALES','ENGLAND','NORTH EAST','NORTH WEST','YORKSHIRE AND THE HUMBER','EAST MIDLANDS','WEST MIDLANDS','SOUTH EAST','WALES','EAST','Tyne and Wear (Met County)','Cumbria','Greater Manchester (Met County)','Lancashire','Merseyside (Met County)','North Yorkshire','South Yorkshire (Met County)','West Yorkshire (Met County)','Derbyshire','Leicestershire','Lincolnshire','Northamptonshire','Nottinghamshire','Staffordshire','Warwickshire','West Midlands (Met County)','Worcestershire','Cambridgeshire','Essex','Hertfordshire','Norfolk','Suffolk','LONDON','Inner London','Outer London','Buckinghamshire','East Sussex','Hampshire','Kent','Oxfordshire','Surrey','West Sussex','SOUTH WEST','Devon','Dorset','Gloucestershire','Somerset']


    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):  
            test_df = pd.read_csv(file_path, encoding="latin1")
            band_condition = (test_df['Band'] == 'All')
            area_condition = (~test_df['Area Name'].isin(excluded_areas))
            processed_df = test_df[band_condition & area_condition].drop(columns=['Band','Not Known6','All'])
            processed_df.to_csv('./LA_bedroom_properties/'+filename, index=False)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
