import marimo

__generated_with = "0.14.16"
app = marimo.App(width="medium")


@app.cell
def _():
    # map_rent_costs.py
    #
    # Merge PIPR cost/m² data onto LAD boundaries and plot choropleth.
    # 1) Merge by code (LAD24CD ↔ Area code)
    # 2) Fallback: robust exact name match (strip ALL whitespace + lower) using:
    #     LAD24NM ↔ Area name, and (if present) LAD24NM ↔ Area Name (rents)
    # Print unmatched (excluding Scotland "S…" and Northern Ireland "N…"). Exclude those LADs from plot.
    #
    # Run:
    #   python map_rent_costs.py

    import os
    import fiona
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from shapely.geometry import shape, Polygon, MultiPolygon
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection
    import matplotlib as mpl

    # ----------------------
    # Config
    # ----------------------
    SHP_PATH = "LA_boundaries/LAD_MAY_2024_UK_BFE.shp"
    CSV_PATH = "outputs/pipr_cost_per_m2.csv"
    OUT_PATH = "outputs/cpm2_overall_map.png"

    # ----------------------
    # Load shapefile
    # ----------------------
    records = []
    with fiona.open(SHP_PATH, "r") as src:
        for feat in src:
            props = dict(feat["properties"])
            geom = shape(feat["geometry"]) if feat["geometry"] else None
            props["__geom__"] = geom
            records.append(props)

    gdf_like = pd.DataFrame(records)

    # ----------------------
    # Load CSV
    # ----------------------
    df = pd.read_csv(CSV_PATH)
    gdf_like.columns = [c.strip() for c in gdf_like.columns]
    df.columns = [c.strip() for c in df.columns]

    # Ensure expected columns exist
    for col in ["LAD24CD", "LAD24NM"]:
        if col not in gdf_like.columns:
            raise KeyError(f"Shapefile missing expected column: {col}")
    if "Area code" not in df.columns:
        raise KeyError("CSV missing expected column: 'Area code'")
    if "Area name" not in df.columns:
        raise KeyError("CSV missing expected column: 'Area name'")

    # ----------------------
    # Primary merge: by code
    # ----------------------
    gdf_like["LAD24CD"] = gdf_like["LAD24CD"].astype(str).str.strip()
    df["Area code"] = df["Area code"].astype(str).str.strip()

    merged = gdf_like.merge(df, left_on="LAD24CD", right_on="Area code", how="left", suffixes=("", "_pipr"))

    # ----------------------
    # Fallback merge: by name (ROBUST exact: strip ALL whitespace, lower())
    # Only for rows still unmatched after code-merge
    # ----------------------
    def _norm_series(s: pd.Series) -> pd.Series:
        # remove all whitespace characters then lowercase
        return s.astype(str).str.replace(r"\s+", "", regex=True).str.lower()

    # Build normalized name keys
    merged["__name_key__"] = _norm_series(merged["LAD24NM"])
    df["__name_key_area__"] = _norm_series(df["Area name"])
    if "Area Name (rents)" in df.columns:
        df["__name_key_rents__"] = _norm_series(df["Area Name (rents)"])
    else:
        df["__name_key_rents__"] = pd.Series(index=df.index, dtype="object")

    # Columns we may want to pull from PIPR on fallback
    value_cols_area = ["cpm2_1", "cpm2_2", "cpm2_3", "cpm2_4plus", "cpm2_overall", "Area code", "Area name"]
    value_cols_rents = ["cpm2_1", "cpm2_2", "cpm2_3", "cpm2_4plus", "cpm2_overall", "Area code", "Area Name (rents)"]

    # Build lookup dicts keyed on normalized names
    area_lookup = df.drop_duplicates("__name_key_area__").set_index("__name_key_area__")
    rents_lookup = (
        df.drop_duplicates("__name_key_rents__").set_index("__name_key_rents__")
        if df["__name_key_rents__"].notna().any() else None
    )

    # Subset of still-unmatched rows
    unmatched_mask = merged["cpm2_overall"].isna()

    # 2a) Try LAD24NM ↔ Area name
    if unmatched_mask.any():
        key_series = merged.loc[unmatched_mask, "__name_key__"]
        for col in value_cols_area:
            if col in area_lookup.columns:
                merged.loc[unmatched_mask, col] = merged.loc[unmatched_mask, col].combine_first(
                    key_series.map(area_lookup[col])
                )
        unmatched_mask = merged["cpm2_overall"].isna()

    # 2b) Optional second fallback: LAD24NM ↔ Area Name (rents)
    if unmatched_mask.any() and (rents_lookup is not None):
        key_series = merged.loc[unmatched_mask, "__name_key__"]
        for col in value_cols_rents:
            if col in rents_lookup.columns:
                merged.loc[unmatched_mask, col] = merged.loc[unmatched_mask, col].combine_first(
                    key_series.map(rents_lookup[col])
                )

    # ----------------------
    # Report missing matches (exclude Scotland + Northern Ireland)
    # ----------------------
    mask_excluded = merged["LAD24CD"].str.upper().str.startswith(("S", "N"))
    missing = merged.loc[merged["cpm2_overall"].isna() & ~mask_excluded, "LAD24NM"].tolist()

    if missing:
        print("Local Authorities with no match in CSV (excluding Scotland + N. Ireland):")
        for nm in missing:
            print("  -", nm)
    else:
        print("All LAD24CD matched successfully after robust fallback (excluding Scotland + N. Ireland).")

    # ----------------------
    # Exclude Scotland + N. Ireland from plotting
    # ----------------------
    merged = merged.loc[~mask_excluded].copy()

    # ----------------------
    # Prepare for plotting
    # ----------------------
    merged["cpm2_overall"] = pd.to_numeric(merged["cpm2_overall"], errors="coerce")
    merged = merged.dropna(subset=["cpm2_overall"])

    patches, values = [], []
    for _, row in merged.iterrows():
        geom = row["__geom__"]
        if geom is None:
            continue

        def add_polygon(poly: Polygon):
            x, y = poly.exterior.coords.xy
            coords = np.column_stack([x, y])
            patches.append(MplPolygon(coords, closed=True))
            values.append(row["cpm2_overall"])

        if isinstance(geom, Polygon):
            add_polygon(geom)
        elif isinstance(geom, MultiPolygon):
            for poly in geom.geoms:
                add_polygon(poly)

    # ----------------------
    # Plot choropleth
    # ----------------------
    def plot_map(patches, values, cmap_name, output_path):
        """Generates and saves a choropleth map with a specified colormap."""
        # Fixed scale for comparability between maps
        norm = mpl.colors.Normalize(vmin=5, vmax=45)
        cmap = mpl.cm.get_cmap(cmap_name)
        colors = cmap(norm(values))

        fig, ax = plt.subplots(figsize=(10, 12))
        pc = PatchCollection(patches, facecolor=colors, edgecolor="black", linewidths=0.2)
        ax.add_collection(pc)
        ax.autoscale_view()
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title("Private Rent Cost per m² (Overall) by Local Authority", pad=12)

        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("£ per m²")

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved map to: {output_path}")


    if values:
        # --- Plot 1: Original color scheme ---
        plot_map(patches, values, cmap_name='plasma', output_path=OUT_PATH)

        # --- Plot 2: RdYlGn_r color scheme ---
        base, ext = os.path.splitext(OUT_PATH)
        out_path_rdygn = f"{base}_RdYlGn_r{ext}"
        plot_map(patches, values, cmap_name='RdYlGn_r', output_path=out_path_rdygn)

    else:
        print("No polygons to plot after merging.")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
