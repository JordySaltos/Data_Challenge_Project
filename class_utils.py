"""Helper functions."""

# Imports

import gc
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from tqdm.auto import tqdm

SEED, TARGET = 3407, "accepted"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Features

SELECTED_VARS = [
    "debt_to_income_ratio", "initially_payable_to_institution", "aus-1", "preapproval", "derived_loan_product_type",
    "ltv_calc", "property_value", "loan_to_value_ratio", "ltv_gap", "state_code", "applicant_race-1", "dti_num",
    "derived_dwelling_category", "loan_term", "applicant_credit_score_type", "income_nonpos", "co-applicant_race-1",
    "monthly_pay_proxy", "units_share", "income_to_msa", "submission_of_application", "applicant_age", "income",
    "pop_per_home", "manufactured_home_secured_property_type", "lti", "open-end_line_of_credit", "pay_to_income",
    "total_units", "co-applicant_sex", "tract_to_msa_income_percentage", "n_missing", "occupancy_type",
    "applicant_ethnicity-1"]
CAT_CODES = ["loan_type", "lien_status", "preapproval", "reverse_mortgage", "open-end_line_of_credit",
             "business_or_commercial_purpose", "negative_amortization", "interest_only_payment", "balloon_payment",
             "other_nonamortizing_features", "construction_method", "occupancy_type",
             "manufactured_home_secured_property_type", "manufactured_home_land_property_interest",
             "applicant_credit_score_type", "co-applicant_credit_score_type", "applicant_ethnicity-1",
             "co-applicant_ethnicity-1", "applicant_race-1", "co-applicant_race-1", "applicant_sex", "co-applicant_sex",
             "submission_of_application", "initially_payable_to_institution", "aus-1"]
CAT_TEXT = ["state_code", "conforming_loan_limit", "derived_loan_product_type", "derived_dwelling_category"]
AGE_ORDER = ["<25", "25-34", "35-44", "45-54", "55-64", "65-74", ">74"]
DTI_MAP = {"<20%": 10, "20%-<30%": 25, "30%-<36%": 33, "50%-60%": 55, ">60%": 65}
MISS_COLS = ["loan_to_value_ratio", "property_value", "income", "loan_term", "county_code", "total_units",
             "debt_to_income_ratio", "applicant_age", "state_code"]

# Preprocessing

class Cleaner:
    """Clips outliers and builds engineered features."""
    def fit(self, d):
        """Learn the 99.9th-percentile clip limits."""
        self.clip_ = {c: d[c].quantile(0.999) for c in ["loan_amount", "property_value"]}
        return self
    def transform(self, r):
        """Return a cleaned copy of a dataset with engineered features."""
        d = r.copy()
        d["n_missing"] = r[MISS_COLS].isna().sum(axis=1)
        d["income_nonpos"] = (r["income"] <= 0).astype(int)
        d["loan_to_value_ratio"] = r["loan_to_value_ratio"].clip(0, 200)
        d["income"] = r["income"].where(r["income"] > 0).clip(upper=5000)
        d["property_value"] = r["property_value"].clip(upper=self.clip_["property_value"])
        d["loan_amount"] = r["loan_amount"].clip(upper=self.clip_["loan_amount"])
        d["loan_term"] = r["loan_term"].clip(upper=480)
        dti = r["debt_to_income_ratio"].astype(object)
        d["dti_num"] = dti.map(DTI_MAP).fillna(pd.to_numeric(dti, errors="coerce"))
        age = r["applicant_age"].astype(object)
        d["applicant_age"] = age.where(age.isin(AGE_ORDER), "Unknown")
        inc = d["income"] * 1000
        msa = r["ffiec_msa_md_median_family_income"].where(r["ffiec_msa_md_median_family_income"] > 0)
        homes = r["tract_one_to_four_family_homes"].replace(0, np.nan)
        d["ltv_calc"] = (r["loan_amount"] / d["property_value"] * 100).clip(0, 300)
        d["ltv_gap"] = d["loan_to_value_ratio"] - d["ltv_calc"]
        d["lti"] = (d["loan_amount"] / inc).clip(0, 50)
        d["income_to_msa"] = (inc / msa).clip(0, 20)
        d["monthly_pay_proxy"] = d["loan_amount"] / r["loan_term"].replace(0, np.nan)
        d["pay_to_income"] = d["monthly_pay_proxy"] / (inc / 12)
        d["units_share"] = r["tract_owner_occupied_units"] / homes
        d["pop_per_home"] = r["tract_population"] / homes
        return d.replace([np.inf, -np.inf], np.nan)
        
class Preprocessor:
    """Cleaning and encoding fitted: produces a "scl" view (imputed, scaled, one-hot)
    and an "ord" view (unscaled numerics with NaN, then ordinal-coded categoricals)."""
    def __init__(self, variables=SELECTED_VARS):
        self.vars = variables
        self.cat = [c for c in variables if c in CAT_CODES + CAT_TEXT + ["debt_to_income_ratio", "applicant_age"]]
        self.num = [c for c in variables if c not in self.cat]
    def _frame(self, d):
        """Clean a dataset, keep the selected variables and cast categoricals to str."""
        d = self.cleaner.transform(d)[self.vars].copy()
        d[self.cat] = d[self.cat].astype(str)
        return d
    def fit(self, d):
        """Fit the cleaner and the encoders."""
        self.cleaner = Cleaner().fit(d)
        f = self._frame(d)
        self.ct = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), self.num),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005, sparse_output=False),
             self.cat)]).fit(f)
        self.oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan).fit(f[self.cat])
        return self
    @property
    def ord_cat_idx(self):
        """Categorical column positions in the "ord" view."""
        return list(range(len(self.num), len(self.num) + len(self.cat)))
    def _scl(self, f):
        """Encode a cleaned frame as the "scl" view."""
        return np.clip(self.ct.transform(f), -10, 10).astype(np.float32)
    def _ord(self, f):
        """Encode a cleaned frame as the "ord" view."""
        num = f[self.num].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
        return np.hstack([num, self.oe.transform(f[self.cat]).astype(np.float32)])
    def transform(self, d):
        """Return the "scl" view."""
        return self._scl(self._frame(d))
    def transform_views(self, d):
        """Return {"scl": array, "ord": array}."""
        f = self._frame(d)
        return {"scl": self._scl(f), "ord": self._ord(f)}
        
# Classic Models

class SeedBag:
    """Average of HistGradientBoosting model."""
    def __init__(self, params, n_seeds=3):
        self.m = [HistGradientBoostingClassifier(**params, random_state=SEED + s) for s in range(n_seeds)]
    def fit(self, X, y):
        """Fit every model."""
        for m in self.m:
            m.fit(X, y)
        return self
    def predict_proba(self, X):
        """Return the mean class probabilities."""
        return np.mean([m.predict_proba(X) for m in self.m], axis=0)
        
class EarlyStopped:
    """Adds early stopping to a booster using an internal 10% validation split."""
    def __init__(self, model, fit_kw=lambda: {}):
        self.model, self.fit_kw = model, fit_kw
    def fit(self, X, y):
        """Fit with the inner split as eval set."""
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.1, stratify=y, random_state=SEED)
        self.model.fit(Xa, ya, eval_set=[(Xb, yb)], **self.fit_kw())
        return self
    def predict_proba(self, X):
        """Return class probabilities."""
        return self.model.predict_proba(X)
        
class TorchMLP:
    """Multi-layer perceptron of Linear-BatchNorm-Sigmoid-Dropout blocks; Adam with early stopping, best epoch restored."""
    def __init__(self, hidden=(128, 64), dropout=0.1, lr=1e-3, batch=512, max_epochs=100, patience=10,
                 val_frac=0.1, seed=SEED):
        self.hidden, self.dropout, self.lr, self.batch = hidden, dropout, lr, batch
        self.max_epochs, self.patience, self.val_frac, self.seed = max_epochs, patience, val_frac, seed
    def _net(self, d):
        """Build the network for the input features."""
        layers, prev = [], d
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.Sigmoid(), nn.Dropout(self.dropout)]
            prev = h
        return nn.Sequential(*layers, nn.Linear(prev, 1))
    def fit(self, X, y):
        """Train with early stopping on an inner validation split."""
        torch.manual_seed(self.seed)
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=self.val_frac, stratify=y, random_state=self.seed)
        t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
        Xa, Xb, ya, yb = t(Xa), t(Xb), t(ya), t(yb)
        self.net = self._net(X.shape[1]).to(DEVICE)
        opt, loss_fn = torch.optim.Adam(self.net.parameters(), lr=self.lr), nn.BCEWithLogitsLoss()
        best, self.best_epoch_, best_state = np.inf, 0, None
        bar = tqdm(range(1, self.max_epochs + 1), unit="epoch")
        for ep in bar:
            self.net.train()
            perm, total = torch.randperm(len(Xa), device=DEVICE), 0.0
            for i in range(0, len(Xa), self.batch):
                idx = perm[i:i + self.batch]
                if len(idx) < 2:
                    continue
                opt.zero_grad()
                loss = loss_fn(self.net(Xa[idx]).squeeze(1), ya[idx])
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            self.net.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.net(Xb).squeeze(1), yb).item()
            if val_loss < best - 1e-5:
                best, self.best_epoch_ = val_loss, ep
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            bar.set_description(f"Epoch {ep}/{self.max_epochs}")
            bar.set_postfix(loss=f"{total / len(Xa):.4f}", val_loss=f"{val_loss:.4f}", best_epoch=self.best_epoch_)
            if ep - self.best_epoch_ >= self.patience:
                break
        bar.close()
        self.net.load_state_dict(best_state)
        return self
    def predict_proba(self, X):
        """Return class probabilities."""
        self.net.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.net(torch.tensor(X, dtype=torch.float32, device=DEVICE)).squeeze(1)).cpu().numpy()
        return np.column_stack([1 - p, p])
        
# Foundation Models

class InContextModel:
    """Base class for in-context models that predict without training."""
    view = "ord"
    def __init__(self, cat_idx=None, max_context=100_000, batch_rows=100_000, seed=SEED):
        self.cat_idx, self.max_context, self.batch_rows, self.seed = list(cat_idx or []), max_context, batch_rows, seed
        self.m = None
    def fit(self, X, y):
        """Store the context and load the model."""
        X, y = np.asarray(X, np.float32), np.asarray(y)
        if self.max_context and len(X) > self.max_context:
            X, _, y, _ = train_test_split(X, y, train_size=self.max_context, stratify=y, random_state=self.seed)
        self.X_, self.y_ = X, y
        self._load()
        return self
    def predict_proba(self, X):
        """Return class probabilities, scored in batches."""
        X = np.asarray(X, np.float32)
        p1 = np.concatenate([np.asarray(self._predict(X[i:i + self.batch_rows]), dtype=np.float64)
                             for i in range(0, len(X), self.batch_rows)])
        return np.column_stack([1 - p1, p1])
    def release(self):
        """Free the model and GPU memory."""
        self.m = self.xc = self.yc = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    def _frame(self, X):
        """Return a DataFrame with categorical columns typed as 'category'."""
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        for i in self.cat_idx:
            df[f"f{i}"] = df[f"f{i}"].astype("category")
        return df
    def _load(self):
        """Build the model."""
        raise NotImplementedError
    def _predict(self, X):
        """Return P(class 1) for a batch."""
        raise NotImplementedError
        
class TabPFNModel(InContextModel):
    """TabPFN, a pretrained transformer for tabular classification."""
    def __init__(self, version="3.5", size="small", n_estimators=None, **kw):
        super().__init__(**kw)
        self.version, self.size, self.n_estimators = version, size, n_estimators
    def _load(self):
        """Create and fit the TabPFN classifier."""
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        v = {"2.6": ModelVersion.V2_6,
             "3.5": ModelVersion.V3_5_FAST if self.size == "small" else ModelVersion.V3_5}[self.version]
        opts = dict(device=DEVICE, categorical_features_indices=self.cat_idx or None,
                    ignore_pretraining_limits=True, random_state=self.seed)
        if self.n_estimators:
            opts["n_estimators"] = self.n_estimators
        self.m = TabPFNClassifier.create_default_for_version(v, **opts)
        self.m.fit(self.X_, self.y_)
    def _predict(self, X):
        """Return P(class 1)."""
        return self.m.predict_proba(X)[:, 1]
        
class TabICLModel(InContextModel):
    """TabICLv2, an in-context model for tabular classification."""
    def __init__(self, n_estimators=8, **kw):
        super().__init__(**kw)
        self.n_estimators = n_estimators
    def _load(self):
        """Create and fit the TabICL classifier."""
        from tabicl import TabICLClassifier
        self.m = TabICLClassifier(n_estimators=self.n_estimators, kv_cache=True, device=DEVICE, random_state=self.seed,
                                  checkpoint_version="tabicl-classifier-v2-20260212.ckpt")
        self.m.fit(self._frame(self.X_), self.y_)
    def _predict(self, X):
        """Return P(class 1)."""
        return self.m.predict_proba(self._frame(X))[:, 1]
        
class KumoTabularModel(InContextModel):
    """Kumo-Tabular, an in-context model for tabular classification."""
    def __init__(self, size="small", num_estimators=8, **kw):
        super().__init__(**kw)
        self.size, self.num_estimators = size, num_estimators
    def _table(self, X, y=None):
        """Convert arrays to an sdm TableTensor, with an optional target column."""
        import sdm
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        stypes = {c: "numerical" for c in df.columns}
        if y is not None:
            df["target"], stypes["target"] = y, "categorical"
        return sdm.TableTensor.from_pandas(df=df, stypes=stypes, device=torch.device(DEVICE))
    def _load(self):
        """Create the model and build the context tables."""
        import sdm
        self.m = sdm.models.KumoTabular(task="classification", size=self.size, device=DEVICE)
        ctx = self._table(self.X_, self.y_)
        self.xc, self.yc = ctx.drop_columns("target"), ctx[:, "target"]
    def _predict(self, X):
        """Return P(class 1)."""
        dev = torch.device(DEVICE)
        with torch.amp.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            probs = self.m(x_context=self.xc, y_context=self.yc, x_query=self._table(X),
                           num_estimators=self.num_estimators)
        pdf = probs.to_pandas()
        return pdf[next((c for c in pdf.columns if str(c) in ("1", "1.0", "True")), pdf.columns[-1])].to_numpy()
        
# Training/Evaluation

def take(X, idx):
    """Select rows from an array or a dict of views."""
    return {k: v[idx] for k, v in X.items()} if isinstance(X, dict) else X[idx]
    
def stack_views(A, B):
    """Concatenate rows of two arrays or two dicts of views."""
    return {k: np.vstack([A[k], B[k]]) for k in A} if isinstance(A, dict) else np.vstack([A, B])
    
def _view(model, X):
    """Select the feature view the model requires (default "scl")."""
    return X[getattr(model, "view", "scl")] if isinstance(X, dict) else X
    
def fit_predict(model, X_train, y_train, X_eval):
    """Fit *model* and return P(accepted=1) for *X_eval*."""
    p = model.fit(_view(model, X_train), y_train).predict_proba(_view(model, X_eval))[:, 1]
    if hasattr(model, "release"):
        model.release()
    return p
    
def run_model(name, model, models, P, X_train, y_train, X_eval, raise_errors=False):
    """Fit *model*, storing it in *models* and its probabilities in *P*."""
    try:
        P[name] = fit_predict(model, X_train, y_train, X_eval)
        models[name] = model
    except Exception as e:
        if raise_errors:
            raise
        print(f"[{name}] skipped: {type(e).__name__}: {e}")
        
def best_threshold(y_true, p):
    """Return the threshold in [0.05, 0.95] that maximizes MCC."""
    grid = np.linspace(0.05, 0.95, 91)
    return grid[np.argmax([matthews_corrcoef(y_true, p >= t) for t in grid])]
    
def report(y_true, p, thr=0.5):
    """Return ranking and thresholded metrics."""
    pred = p >= thr
    return {"AUC": roc_auc_score(y_true, p),
            "PR-AUC (rejected)": average_precision_score(1 - y_true, 1 - p),
            "MCC": matthews_corrcoef(y_true, pred),
            "F1 macro": f1_score(y_true, pred, average="macro"),
            "Bal. acc.": balanced_accuracy_score(y_true, pred),
            "Accuracy": (pred == y_true).mean()}
    
def results_table(P, y_true):
    """Return metrics table sorted by AUC."""
    thr = {n: best_threshold(y_true, p) for n, p in P.items()}
    table = pd.DataFrame({n: {**report(y_true, p, thr[n]), "Threshold": thr[n]} for n, p in P.items()}).T
    return table.sort_values("AUC", ascending=False).round(4), thr
    
def refit_predict(models, best, X, y, X_new):
    """Refit the selected model on (X, y) and predict *X_new*."""
    return fit_predict(models[best], X, y, X_new)