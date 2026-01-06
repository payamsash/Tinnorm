'''
## create deviation topoplots by number of subjects deviating z-thr per freq band
## same for source maps
## same for conn map
## same for aperiodic

## correlate deviations with clinical outcomes



## per connection, how many subjects deviated (skip region level and network level)
## per power
## aperiodic
'''

import pandas as pd
from pathlib import Path
from scipy.stats import ranksums
from statsmodels.stats.multitest import fdrcorrection

df_test = pd.read_csv(f"/Users/payamsadeghishabestari/Tinnorm/material/test_model_results/results/Z_test.csv")
df_train = pd.read_csv(f"/Users/payamsadeghishabestari/Tinnorm/material/test_model_results/results/Z_train.csv")
z_thr = 2.3

df_test = df_test.filter(regex=r"alpha_1")
df_train = df_train.filter(regex=r"alpha_1")

test_pos = (df_test > z_thr).astype(int)
test_neg = (df_test < -z_thr).astype(int)

train_pos = (df_train > z_thr).astype(int)
train_neg = (df_train < -z_thr).astype(int)

def wilcoxon_per_region(test_df, train_df):
    """Wilcoxon rank-sum test per column."""
    return pd.Series(
        {
            col: ranksums(test_df[col], train_df[col]).pvalue
            for col in test_df.columns
        }
    )

pvals_pos = wilcoxon_per_region(test_pos, train_pos)
pvals_neg = wilcoxon_per_region(test_neg, train_neg)

summary = pd.DataFrame(
    {
        "test_pos_mean": test_pos.mean(axis=0),
        "train_pos_mean": train_pos.mean(axis=0),
        "test_neg_mean": test_neg.mean(axis=0),
        "train_neg_mean": train_neg.mean(axis=0),
        "p_pos": pvals_pos,
        "p_neg": pvals_neg,
    }
)
summary["p_pos_fdr"] = fdrcorrection(summary["p_pos"])[1]
summary["p_neg_fdr"] = fdrcorrection(summary["p_neg"])[1]



