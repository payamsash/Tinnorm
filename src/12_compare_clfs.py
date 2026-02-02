from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import (
    GroupKFold,
    StratifiedGroupKFold
    )
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from dataclasses import dataclass
from typing import Optional

@dataclass
class ClassificationResult:
    df_metric: pd.DataFrame
    y: Optional[np.ndarray] = None
    y_prob_1: Optional[np.ndarray] = None
    y_prob_2: Optional[np.ndarray] = None
    delta_null: Optional[np.ndarray] = None
    real_delta: Optional[np.ndarray] = None
    p_value: Optional[float] = None
    y1_folds: Optional[np.ndarray] = None
    p1_folds: Optional[np.ndarray] = None
    y2_folds: Optional[np.ndarray] = None
    p2_folds: Optional[np.ndarray] = None


def _read_the_file(
                tinnorm_dir,
                data_mode,
                mode,
                space,
                freq_band,
                preproc_level,
                conn_mode,
                thi_threshold=None
                ):
    
    hm_dir = tinnorm_dir / "harmonized"
    models_dir = tinnorm_dir / "models"
    df_master = pd.read_csv("../material/master.csv")

    ## Residual data
    if data_mode == "residual":
        mode_prefix = ""
        if mode in ["conn", "global", "regional", "diffusive"]:
            mode_prefix += f"_{conn_mode}"

        fname = hm_dir / f"preproc_{preproc_level}" / space / f"{mode}{mode_prefix}_residual.csv"
        df = pd.read_csv(fname)

        if not thi_threshold is None:
            df = df.merge(
                        df_master[['subject_id', 'thi_score']],
                        on='subject_id',
                        how='left'
                    )
            df = df.query(f'group == 0 or thi_score > {thi_threshold}')

        cols_to_drop = ["Unnamed: 0", "subject_id", "age", "sex", "PTA4_mean", "thi_score"]
        df.drop(columns=cols_to_drop, inplace=True, errors="ignore")
        
    ## Deviation data
    if data_mode == "deviation":
        mode_prefix = ""
        if mode in ["conn", "global", "regional", "diffusive"]:
            mode_prefix += f"_{conn_mode}"
        
        dfs_group = []
        for group_name, group_id in zip(["train", "test"], [0, 1]):
            fname = (
                        models_dir
                        / f"preproc_{preproc_level}"
                        / space
                        / f"{mode}{mode_prefix}"
                        / "full_model"
                        / "results"
                        / f"Z_{group_name}.csv"
                    )
            df_group = pd.read_csv(fname)
            df_group["group"] = group_id
            dfs_group.append(df_group)

        df = pd.concat(dfs_group, axis=0, ignore_index=True)
        df.rename(columns={"subject_ids": "subject_id"}, inplace=True)

        df = df.merge(df_master[['subject_id', 'site']], on='subject_id', how='left')
        df.rename(columns={"site": "SITE"}, inplace=True)

        if not thi_threshold is None:
            df = df.merge(
                        df_master[['subject_id', 'thi_score']],
                        on='subject_id',
                        how='left'
                    )
            df = df.query(f'group == 0 or thi_score > {thi_threshold}')

        df.sort_values("subject_id", inplace=True)
        df.drop(columns=["observations", "subject_id", "thi_score"], inplace=True, errors="ignore")
    
    y = df["group"].to_numpy()         
    sites = df["SITE"].to_numpy()

    print("\n****************************************************")
    print(f"number of subjects are: {df['group'].value_counts()}")
    print("****************************************************\n")

    if not freq_band is None:
        df = df.filter(regex=rf"{freq_band}")
    
    X = df[list(df.columns)[:-2]]
    print("\n****************************************************")
    print(f"number of selected features are: {X.shape[1]}")
    print("****************************************************\n")

    return X, y, sites


def _initiate_ml_model(ml_model, n_jobs):

    kwargs = {
        "class_weight": "balanced",
        "random_state" : 42
    }

    if ml_model == "RF":
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=5,
            n_jobs=n_jobs,
            **kwargs
        )
    if ml_model == "SVM":
        model = SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            **kwargs
        )
    
    return model

def _initiate_rfe_model(base_model, n_features_to_select=None, step=1):
    """
    Wrap a classifier with RFE. If n_features_to_select is None, defaults to half of features.
    """
    rfe_model = RFE(
                    estimator=base_model,
                    n_features_to_select=n_features_to_select,
                    step=step,
                    verbose=2
                    )
    return rfe_model

def _run_permutation(X_np, y, sites, model, ml_model, n_permutations, folding_mode):

    y_pred, y_prob = run_cv(X_np, y, sites, model, folding_mode)
    df_metric = metrics_to_dataframe(
        model_name=f"{ml_model}_real",
        y=y,
        y_pred=y_pred,
        y_prob=y_prob
    )

    rng = np.random.default_rng(42)
    print("*************** running permutation tests ***************")
    for i in tqdm(range(n_permutations)):
        y_perm = rng.permutation(y)
        y_pred_p, y_prob_p = run_cv(X_np, y_perm, sites, model, folding_mode)
        df_p = metrics_to_dataframe(
            model_name=f"{ml_model}_perm_{i+1}",
            y=y_perm,
            y_pred=y_pred_p,
            y_prob=y_prob_p
        )
        df_metric = pd.concat([df_metric, df_p], ignore_index=True)
    
    return df_metric

def metrics_to_dataframe(model_name, y, y_pred, y_prob=None):
    """Return a DataFrame with common classification metrics for a model."""

    bal_acc = balanced_accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y, y_pred, average="weighted", zero_division=0)
    
    # ROC-AUC: only if probabilities or decision function is provided
    if y_prob is not None:
        try:
            auc = roc_auc_score(y, y_prob, multi_class="ovo", average="weighted")
        except ValueError:
            auc = None
    else:
        auc = None

    return pd.DataFrame({
        "model": [model_name],
        "balanced_accuracy": [bal_acc],
        "precision": [prec],
        "recall": [rec],
        "f1-score": [f1],
        "roc_auc": [auc]
    })

def run_cv(X, y, sites, model, folding_mode):

    if folding_mode == "loso":
        splitter = GroupKFold(n_splits=len(np.unique(sites)))

    elif folding_mode == "stratified":
        splitter = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

    else:
        raise ValueError("folding_mode must be 'loso' or 'stratified'")

    y_pred = np.zeros_like(y)
    y_prob = np.zeros(len(y))

    for train_idx, test_idx in splitter.split(X, y, groups=sites):
        model.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = model.predict(X[test_idx])
        y_prob[test_idx] = model.predict_proba(X[test_idx])[:, 1]

    return y_pred, y_prob


def run_loso_cv_with_folds(X, y, sites, model):
    gkf = GroupKFold(n_splits=len(np.unique(sites)))
    
    fold_probs = []
    fold_true = []

    for train_idx, test_idx in gkf.split(X, y, groups=sites):
        model.fit(X[train_idx], y[train_idx])
        prob = model.predict_proba(X[test_idx])[:, 1]
        fold_probs.append(prob)
        fold_true.append(y[test_idx])

    return fold_true, fold_probs


def _run_comparison(X_1, X_2, y, y_2, sites, sites_2, model, n_permutations, folding_mode):

    # checks
    if not np.array_equal(y, y_2):
        raise ValueError("y and y_2 are not identical. Subject labels do not match.")

    if not np.array_equal(sites, sites_2):
        raise ValueError("sites and sites_2 are not identical. LOSO grouping does not match.")

    if X_1.shape[0] != X_2.shape[0]:
        raise ValueError("X1 and X2 have different number of subjects.")

    print("y and sites match across residual and deviation data.")

    # Run LOSO-CV for both feature sets 
    y_pred_1, y_prob_1 = run_cv(X_1, y, sites, model, folding_mode)
    y_pred_2, y_prob_2 = run_cv(X_2, y, sites, model, folding_mode)

    # Metrics tables
    df_X1 = metrics_to_dataframe("X1", y, y_pred_1, y_prob_1)
    df_X2 = metrics_to_dataframe("X2", y, y_pred_2, y_prob_2)

    df_metric = pd.concat([df_X1, df_X2], ignore_index=True)

    # Real performance difference
    auc_1 = roc_auc_score(y, y_prob_1)
    auc_2 = roc_auc_score(y, y_prob_2)
    real_delta = auc_1 - auc_2

    print(f"Real AUC X1: {auc_1:.3f}")
    print(f"Real AUC X2: {auc_2:.3f}")
    print(f"ΔAUC (X1 - X2): {real_delta:.3f}")

    # Paired permutation test
    delta_null = np.zeros(n_permutations)
    rng = np.random.default_rng(42)

    for i in range(n_permutations):
        swap = rng.random(len(y)) < 0.5
        y1_p = y_prob_1.copy()
        y2_p = y_prob_2.copy()

        y1_p[swap], y2_p[swap] = y2_p[swap], y1_p[swap]
        auc1_p = roc_auc_score(y, y1_p)
        auc2_p = roc_auc_score(y, y2_p)
        delta_null[i] = auc1_p - auc2_p

    p_value = np.mean(np.abs(delta_null) >= np.abs(real_delta))
    print(f"Permutation p-value (X1 vs X2): {p_value:.4f}")

    return  df_metric, y, y_prob_1, y_prob_2, delta_null, real_delta, p_value

def save_clf_result(res, clfs_dir, run_mode):

    clfs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = clfs_dir / run_mode / f"metrics_{timestamp}.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    res.df_metric.to_csv(metrics_path, index=False)

    # comparison
    if res.p_value is None:
        return

    comp_dir = clfs_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    arrays = {
        "y": res.y,
        "y_prob_1": res.y_prob_1,
        "y_prob_2": res.y_prob_2,
        "delta_null": res.delta_null,
        "real_delta": res.real_delta,
        "y1_folds": res.y1_folds,
        "p1_folds": res.p1_folds,
        "y2_folds": res.y2_folds,
        "p2_folds": res.p2_folds,
        "p_value": res.p_value
    }

    for name, arr in arrays.items():
        np.save(comp_dir / f"{name}.npy", arr)

def classify(
        tinnorm_dir,
        data_mode,
        preproc_level,
        space, 
        mode,
        conn_mode,
        freq_band,
        folding_mode,
        high_corr_drop,
        corr_thr,
        ml_model,
        n_jobs,
        run_permutation=False,
        n_permutations=100,
        run_comparison=False,
        apply_rfe=False,
        n_rfe_features=100,
        thi_threshold=None,
    ):

    X, y, sites = _read_the_file(
                                tinnorm_dir,
                                data_mode,
                                mode,
                                space,
                                freq_band,
                                preproc_level,
                                conn_mode,
                                thi_threshold
                                )

    ## remove highly correlated features
    if high_corr_drop:
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > corr_thr)]
        X.drop(columns=to_drop, inplace=True)
        print(f"*** {len(to_drop)} dropped from {X.shape[1]} features ***")
        print(f"*********************************************************")

    model = _initiate_ml_model(ml_model, n_jobs)
    X_1 = X.to_numpy()
    df_metric = pd.DataFrame()

    if apply_rfe:
        n_select = min(n_rfe_features, X_1.shape[1] // 2)
        print(f"Applying RFE: selecting top {n_select} features")
        
        if mode == "conn":
            step = 200
        else:
            step = 20

        model = _initiate_rfe_model(model, n_features_to_select=n_select, step=step)
    
    ## permutation test
    if run_permutation:
        df_metric = _run_permutation(X_1, y, sites, model, ml_model, n_permutations, folding_mode)
        # return df_metric

    # compare features
    if run_comparison:
        if data_mode != "residual":
            raise ValueError(
                f"data_mode must be set to 'residual' for comparison, got '{data_mode}' instead."
            )
        X_2, y_2, sites_2 = _read_the_file(
            tinnorm_dir, "deviation", mode, space, freq_band, preproc_level, conn_mode, thi_threshold
            )
        X_2 = X_2.to_numpy()

        df_metric, y, y_prob_1, y_prob_2, delta_null, \
            real_delta, p_value  = \
                _run_comparison(X_1, X_2, y, y_2, sites, sites_2, model, n_permutations)
        
        ## get CI for the ROC curve by running LOSO with folds
        y1_folds, p1_folds = run_loso_cv_with_folds(X_1, y, sites, model)
        y2_folds, p2_folds = run_loso_cv_with_folds(X_2, y, sites, model)

    # add context columns
    if not df_metric.empty:
        len_df = len(df_metric)
        col_names = ["mode", "space", "preproc_level", "conn_mode", "high_corr_drop", "data_mode", "folding_mode"]
        col_vals = [mode, space, preproc_level, conn_mode, high_corr_drop, data_mode, folding_mode]
        for col_name, col_val in zip(col_names, col_vals):
            df_metric[col_name] = [col_val] * len_df

    if run_comparison: 
        df_metric.at[1, "data_mode"] = "deviation"
        
        return ClassificationResult(
                                    df_metric=df_metric,
                                    y=y,
                                    y_prob_1=y_prob_1,
                                    y_prob_2=y_prob_2,
                                    delta_null=delta_null,
                                    real_delta=real_delta,
                                    p_value=p_value,
                                    y1_folds=y1_folds,
                                    p1_folds=p1_folds,
                                    y2_folds=y2_folds,
                                    p2_folds=p2_folds,
                                )
    return ClassificationResult(df_metric=df_metric)


if __name__ == "__main__":
    
    tinnorm_dir = Path("/Volumes/Extreme_SSD/payam_data/Tinnorm")
    clfs_dir = tinnorm_dir / "clfs"
    kwargs = dict(
                    data_mode = "residual",
                    preproc_level = 2,
                    space = "source",
                    mode = "power",
                    conn_mode = None,
                    freq_band = "alpha_1",
                    folding_mode = "stratified",
                    high_corr_drop = False,
                    corr_thr = 0.95,
                    ml_model = "RF",
                    n_jobs=-1,
                    run_permutation = True,
                    n_permutations = 10,
                    run_comparison = False,
                    apply_rfe = False,
                    n_rfe_features = 100,
                    thi_threshold = None
                    )
    
    ## some checks
    mode = kwargs.get("mode")
    conn_mode = kwargs.get("conn_mode")
    freq_band = kwargs.get("freq_band")
    run_permutation = kwargs.get("run_permutation", False)
    run_comparison = kwargs.get("run_comparison", False)

    valid_modes = ["conn", "regional", "global", "diffusive"]
    if mode not in valid_modes:
        if conn_mode is not None:
            raise ValueError(f"conn_mode must be None when mode='{mode}', got {conn_mode} instead.")

    if mode == "aperiodic" and freq_band is not None:
        raise ValueError(f"freq_band must be None when mode='aperiodic', got {freq_band} instead.")

    if run_permutation == run_comparison:
        raise ValueError("Exactly one of run_permutation or run_comparison must be True.")
    
    if run_permutation:
        run_mode = "permutation"
    elif run_comparison:
        run_mode = "comparison"

    ## the real part!
    res = classify(tinnorm_dir, **kwargs)
    save_clf_result(res, clfs_dir, run_mode)

    ## from simplest to most complicated
    ## must create folders for saving necessary stuff