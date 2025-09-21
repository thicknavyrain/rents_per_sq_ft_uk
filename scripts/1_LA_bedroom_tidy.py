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
    # Load the new data
    df = pd.read_csv("./LA_bedroom_properties/raw/CTSOP3_0_2024_03_31.csv")

    # Filter the data
    laua_condition = df['geography'] == 'LAUA'
    band_condition = df['band'] == 'All'
    filtered_df = df[laua_condition & band_condition].copy()

    # Drop unnecessary columns (keeping 'ecode')
    columns_to_drop = ['geography', 'ba_code', 'band', 'annexe', 'caravan_houseboat_mobilehome', 'unknown', 'all_properties']
    unkw_columns = [col for col in filtered_df.columns if col.endswith('_unkw')]
    columns_to_drop.extend(unkw_columns)
    filtered_df.drop(columns=columns_to_drop, inplace=True)

    # Define property types and their output file names
    property_types = {
        'bungalow': 'bungalow_bedrooms.csv',
        'flat_mais': 'flat_maisonette_bedrooms.csv',
        'house_terraced': 'house_terraced_bedrooms.csv',
        'house_semi': 'house_semi_detached_bedrooms.csv',
        'house_detached': 'house_detached_bedrooms.csv'
    }

    # Create a directory for the output files
    output_directory = './LA_bedroom_properties/'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Process each property type
    for prefix, filename in property_types.items():
        # Select columns for the current property type, now including 'ecode'
        property_columns = ['area_name', 'ecode'] + [col for col in filtered_df.columns if col.startswith(prefix)]
        property_df = filtered_df[property_columns].copy()

        # Drop the total column for the property type
        property_df.drop(columns=[f'{prefix}_total'], inplace=True)

        # Rename bedroom columns, including "6+"
        rename_dict = {f'{prefix}_{i}': str(i) for i in range(1, 6)}
        rename_dict[f'{prefix}_6'] = '6+'
        property_df.rename(columns=rename_dict, inplace=True)

        # Save to CSV
        output_path = os.path.join(output_directory, filename)
        property_df.to_csv(output_path, index=False)
        print(f"Successfully created {output_path}")
    return


@app.cell
def _():
    
    return


if __name__ == "__main__":
    app.run()
