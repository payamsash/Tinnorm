from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GroupKFold,
    cross_val_predict,
    permutation_test_score
)
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def classify(
            modality,
            mode,
            conn_mode,
            harmonized_dir,
            master_dir,
            preproc_level=2,
            high_corr_drop=True,
            corr_thr=0.9,
            run_permutation=False,
            random_state=42,
            n_jobs=-1        
            ):

    ## read df and add group column
    df_resid = pd.read_csv("/Volumes/G_USZ_ORL$/Research/ANT/tinnorm/harmonized/power_sensor_preproc_2_residual.csv")
    df_master = pd.read_csv("../material/master.csv")

    df_resid["subject_id"] = df_resid["subject_id"].astype(str)
    df_master["subject_id"] = df_master["subject_id"].astype(str)

    df_resid = df_resid.merge(
                        df_master[["subject_id", "group"]],
                        on="subject_id",
                        how="inner"
                    )
    df_resid.drop(columns=["Unnamed: 0", "subject_id", "age", "sex"], inplace=True, errors="ignore")


    columns = list(df_resid.columns)
    X = df_resid[columns[:-2]]

    # Identify highly correlated features and remove them
    if high_corr_drop:
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > corr_thr)]

        X.drop(columns=to_drop, inplace=True)
        print(f"{len(to_drop)} dropped from {X.shape[1]} features ...")


    X_np = X.to_numpy()
    y = df_resid["group"].to_numpy()         
    sites = df_resid["SITE"].to_numpy()

    rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=random_state,
    n_jobs=n_jobs
    )

    ## Leave-One-Site-Out (LOSO) CV
    gkf = GroupKFold(n_splits=len(np.unique(sites)))

    y_pred = np.zeros_like(y)
    y_proba = np.zeros(len(y))

    for train_idx, test_idx in gkf.split(X_np, y, groups=sites):
        rf.fit(X_np[train_idx], y[train_idx])
        y_pred[test_idx] = rf.predict(X_np[test_idx])
        y_proba[test_idx] = rf.predict_proba(X_np[test_idx])[:, 1]

    
    ## evaluation
    bal_acc = balanced_accuracy_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)

    print(f"Balanced accuracy: {bal_acc:.3f}")
    print(f"Accuracy: {acc:.3f}")
    print(f"ROC-AUC: {auc:.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred, digits=3))
    
    ## can RF predict site? (should be chance)
    rf_site = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=n_jobs
    )

    site_pred = cross_val_predict(
        rf_site, X, sites, cv=5
    )

    site_acc = accuracy_score(sites, site_pred)
    print(f"Site prediction accuracy: {site_acc:.3f}")

    ## permutation test
    if run_permutation:
        print(f"running permutation tests ...")
        score, perm_scores, pvalue = permutation_test_score(
                                                            rf,
                                                            X,
                                                            y,
                                                            groups=sites,
                                                            cv=gkf,
                                                            scoring="balanced_accuracy",
                                                            n_permutations=1000,
                                                            n_jobs=n_jobs,
                                                            random_state=random_state
                                                        )
        print(f"Permutation p-value: {pvalue:.4f}")


if __name__ == "__main__":

    modality = "power"
    mode = "sensor"
    conn_mode = None
    preproc_level = 2
    harmonized_dir = Path("")
    master_dir = Path("../material/master.csv")

    classify(
            modality,
            mode,
            conn_mode,
            harmonized_dir,
            master_dir,
            preproc_level=2,
            high_corr_drop=True,
            corr_thr=0.9,
            run_permutation=False,
            random_state=42        
            )
    
    ## add extra mode option for PLI or Coh, PLV