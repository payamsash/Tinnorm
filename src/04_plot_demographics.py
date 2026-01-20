import os
from pathlib import Path
import seaborn as sns
import pandas as pd

tinnorm_dir = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
hues = ["group", "sex"]
df = pd.read_csv("../material/master.csv")
df_plot = df[["site", "sex", "age", "group"]]


site_names = df_plot["site"].unique()
print(site_names)
pal = [
        sns.cubehelix_palette(3, rot=-.2, light=.7).as_hex()[1],
        sns.color_palette("ch:s=-.2,r=.6", as_cmap=False).as_hex()[2]
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
        g.add_legend()
        g.set(yticks=[], ylabel="", xlabel=r"age, sex")
        g.despine(bottom=True, left=True)
        g.figure.savefig(saving_dir / f"{hue}_distribution.pdf", 
                        format="pdf",       
                        dpi=300,            
                        bbox_inches="tight"
                        )