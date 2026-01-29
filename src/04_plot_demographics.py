import os
from pathlib import Path
import seaborn as sns
import pandas as pd

## initiate
tinnorm_dir = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
features_dir = tinnorm_dir / "features"
hues = ["group", "sex"]
df = pd.read_csv("../material/master.csv")
df_plot = df[["subject_id", "site", "sex", "age", "group"]]

df_plot["sex"] = df_plot["sex"].map({1: "Male", 2: "Female"})
df_plot["group"] = df_plot["group"].map({0: "Control", 1: "Tinnitus"})

## finding common subjects
check_text = f"power_sensor_preproc_1.zip"

ids_list = set(
                [
                fname.stem[4:9] for fname 
                in features_dir.iterdir() 
                if str(fname).endswith(check_text)
                ]
                )
ids_df = set(df_plot["subject_id"].astype(str))

only_in_list = sorted(ids_list - ids_df)
only_in_df = sorted(ids_df - ids_list)

print(f"Subjects only in list ({len(only_in_list)}):")
print(only_in_list if only_in_list else "None")

print(f"\nSubjects only in dataframe ({len(only_in_df)}):")
print(only_in_df if only_in_df else "None")

counts = (
        pd.crosstab(df_plot["site"], df_plot["group"])
        .rename(columns={0: "control", 1: "patient"})
        )

print("\n************** SITE × GROUP COUNTS **************")
print(f"{counts}")
print("*************************************************\n")

## plotting part
common_ids = ids_list & ids_df
df_plot = df_plot[df_plot["subject_id"].astype(str).isin(common_ids)]
site_names = df_plot["site"].unique()
print(site_names)
pal = [
        sns.cubehelix_palette(3, rot=-.1, light=.2).as_hex()[1],
        sns.color_palette("ch:s=-.2,r=.8", as_cmap=False).as_hex()[2]
]

xlim = [5, 80]
bw_adjust = 0.5
saving_dir = tinnorm_dir / "plots" / "demographics"
os.makedirs(saving_dir, exist_ok=True)

for hue in hues:
        g = sns.FacetGrid(
                df_plot, row="site", hue=hue, aspect=3.5, height=1.6,
                palette=pal, row_order=site_names, xlim=xlim
        )

        g.map(sns.kdeplot, "age", bw_adjust=bw_adjust, clip_on=False, clip=xlim,
                fill=True, alpha=0.7, linewidth=1.5)
        g.map(sns.kdeplot, "age", clip_on=False, color="w", clip=xlim,
                lw=1.5, bw_adjust=bw_adjust)
        g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

        g.figure.subplots_adjust(hspace=.15, top=0.72)
        g.set_titles("")
        g.add_legend(title="")
        g.set(yticks=[], ylabel="", xlabel=r"Age")
        g.despine(bottom=True, left=True)
        g.figure.savefig(saving_dir / f"{hue}_distribution.pdf", 
                        format="pdf",       
                        dpi=300,            
                        bbox_inches="tight"
                        )