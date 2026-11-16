"""Helper functions for Regression."""

# Imports

import gc
import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from tqdm.auto import tqdm

SEED, TARGET = 3407, "TotalKg"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Features

NUM = ["Age", "BodyweightKg", "MeetYear"]
CAT = ["Equipment", "AgeClass", "BirthYearClass", "Country", "State",
           "Federation", "ParentFederation", "MeetCountry", "MeetState", "Sanctioned"]
COUNT_COLS = ["Federation", "Country", "State", "MeetCountry", "MeetState"]
ENGINEERED = ["IsMale", "WeightClass_num", "BW_minus_Class", "BirthYear", "IsJunior",
              "IsMaster", "IsTested", "IsUSA", "SameCountry", "ParentFedUnknown", "StateUnknown",
              "Age_log", "BodyweightKg_log", "BW_x_Male"] + [f"{c}_count" for c in COUNT_COLS]
NUM_VARS = NUM + ENGINEERED
CAT_VARS = CAT

def class_to_num(s):
    """Return the numeric limit of a weight class."""
    m = re.match(r"\d+(\.\d+)?", str(s))
    return float(m.group()) if m else np.nan
    
def target_bins(y, q=10):
    """Return quantile bins of a continuous target."""
    return pd.qcut(y, q=q, labels=False, duplicates="drop")
    
# Preprocessing

class Cleaner:
    """Builds engineered variables from explanatory columns."""
    def fit(self, d):
        """Learn the category frequency tables."""
        self.counts_ = {c: d[c].astype(str).value_counts() for c in COUNT_COLS}
        return self
    def transform(self, r):
        """Return a copy of a dataset with engineered variables."""
        d = r.copy()
        for c in CAT:
            d[c] = d[c].astype(str)
        age, bw = d["Age"], d["BodyweightKg"]
        wc = d["WeightClassKg"].map(class_to_num)
        d["IsMale"] = (d["Sex"] == "M").astype(int)
        d["WeightClass_num"] = wc
        d["WeightClassPlus"] = d["WeightClassKg"].str.endswith("+").astype(int)
        d["BW_minus_Class"] = bw - wc
        d["BirthYear"] = d["MeetYear"] - age
        d["IsJunior"] = (age < 23).astype(int)
        d["IsMaster"] = (age >= 40).astype(int)
        d["IsRaw"] = (d["Equipment"] == "Raw").astype(int)
        d["IsTested"] = (d["Tested"] == "Yes").astype(int)
        d["IsUSA"] = (d["Country"] == "USA").astype(int)
        d["SameCountry"] = ((d["Country"] == d["MeetCountry"]) & (d["Country"] != "Unknown")).astype(int)
        d["ParentFedUnknown"] = (d["ParentFederation"] == "Unknown").astype(int)
        d["StateUnknown"] = (d["State"] == "Unknown").astype(int)
        d["Age_log"] = np.log(age.clip(lower=0.1))
        d["BodyweightKg_log"] = np.log(bw.clip(lower=1))
        d["BW_x_Male"] = bw * d["IsMale"]
        for c in COUNT_COLS:
            d[f"{c}_count"] = d[c].map(self.counts_[c]).fillna(0)
        return d.replace([np.inf, -np.inf], np.nan)
        
class Preprocessor:
    """Cleaning and encoding fitted: produces a "scl" view (imputed, scaled, one-hot)
    and an "ord" view (unscaled numerics with NaN, then ordinal-coded categoricals)."""
    def __init__(self, num=NUM_VARS, cat=CAT_VARS):
        self.num, self.cat = list(num), list(cat)
    def _frame(self, d):
        """Clean a dataset, keep the model variables and cast categoricals to str."""
        d = self.cleaner.transform(d)[self.num + self.cat].copy()
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
        self.m = [HistGradientBoostingRegressor(**params, random_state=SEED + s) for s in range(n_seeds)]
    def fit(self, X, y):
        """Fit every model."""
        for m in self.m:
            m.fit(X, y)
        return self
    def predict(self, X):
        """Return the mean prediction."""
        return np.mean([m.predict(X) for m in self.m], axis=0)
        
class EarlyStopped:
    """Adds early stopping to a booster using an internal 10% validation split."""
    def __init__(self, model, fit_kw=lambda: {}):
        self.model, self.fit_kw = model, fit_kw
    def fit(self, X, y):
        """Fit with the inner split as eval set."""
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.1, random_state=SEED)
        self.model.fit(Xa, ya, eval_set=[(Xb, yb)], **self.fit_kw())
        return self
    def predict(self, X):
        """Return predictions."""
        return self.model.predict(X)
        
class TorchMLP:
    """Multi-layer perceptron of Linear-BatchNorm-Sigmoid-Dropout blocks; Adam with early stopping, best epoch restored."""
    def __init__(self, hidden=(128, 64), dropout=0.1, lr=1e-3, batch=512, max_epochs=100, patience=10,
                 val_frac=0.1, seed=SEED):
        self.hidden, self.dropout, self.lr, self.batch = hidden, dropout, lr, batch
        self.max_epochs, self.patience, self.val_frac, self.seed = max_epochs, patience, val_frac, seed
    def _net(self, d):
        """Build the network for `d` input features."""
        layers, prev = [], d
        for h in self.hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.Sigmoid(), nn.Dropout(self.dropout)]
            prev = h
        return nn.Sequential(*layers, nn.Linear(prev, 1))
    def fit(self, X, y):
        """Train with early stopping on an inner validation split."""
        torch.manual_seed(self.seed)
        self.mu_, self.sd_ = float(np.mean(y)), float(np.std(y))
        z = (np.asarray(y) - self.mu_) / self.sd_
        Xa, Xb, za, zb = train_test_split(X, z, test_size=self.val_frac, random_state=self.seed)
        t = lambda a: torch.tensor(a, dtype=torch.float32, device=DEVICE)
        Xa, Xb, za, zb = t(Xa), t(Xb), t(za), t(zb)
        self.net = self._net(X.shape[1]).to(DEVICE)
        opt, loss_fn = torch.optim.Adam(self.net.parameters(), lr=self.lr), nn.MSELoss()
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
                loss = loss_fn(self.net(Xa[idx]).squeeze(1), za[idx])
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            self.net.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.net(Xb).squeeze(1), zb).item()
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
    def predict(self, X):
        """Return predictions on the original target scale."""
        self.net.eval()
        with torch.no_grad():
            z = self.net(torch.tensor(X, dtype=torch.float32, device=DEVICE)).squeeze(1).cpu().numpy()
        return z * self.sd_ + self.mu_
        
# Foundation Models

class InContextModel:
    """Base class for in-context models that predict without training."""
    view = "ord"
    def __init__(self, cat_idx=None, max_context=50_000, batch_rows=10_000, seed=SEED):
        self.cat_idx, self.max_context, self.batch_rows, self.seed = list(cat_idx or []), max_context, batch_rows, seed
        self.m = None
    def fit(self, X, y):
        """Store the context and load the model."""
        X, y = np.asarray(X, np.float32), np.asarray(y, np.float64)
        if self.max_context and len(X) > self.max_context:
            X, _, y, _ = train_test_split(X, y, train_size=self.max_context, stratify=target_bins(y),
                                          random_state=self.seed)
        self.X_, self.y_ = X, y
        self._load()
        return self
    def predict(self, X):
        """Return predictions, scored in batches."""
        X = np.asarray(X, np.float32)
        return np.concatenate([np.asarray(self._predict(X[i:i + self.batch_rows]), dtype=np.float64)
                               for i in range(0, len(X), self.batch_rows)])
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
        """Return predictions for a batch."""
        raise NotImplementedError
        
class TabPFNModel(InContextModel):
    """TabPFN, a pretrained transformer for tabular regression."""
    def __init__(self, version="3.5", size="small", n_estimators=None, **kw):
        super().__init__(**kw)
        self.version, self.size, self.n_estimators = version, size, n_estimators
    def _load(self):
        """Create and fit the TabPFN regressor."""
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        v = {"2.6": ModelVersion.V2_6,
             "3.5": ModelVersion.V3_5_FAST if self.size == "small" else ModelVersion.V3_5}[self.version]
        opts = dict(device=DEVICE, categorical_features_indices=self.cat_idx or None,
                    ignore_pretraining_limits=True, random_state=self.seed)
        if self.n_estimators:
            opts["n_estimators"] = self.n_estimators
        self.m = TabPFNRegressor.create_default_for_version(v, **opts)
        self.m.fit(self.X_, self.y_)
    def _predict(self, X):
        """Return the mean of the predictive distribution."""
        return self.m.predict(X)
        
class TabICLModel(InContextModel):
    """TabICLv2, an in-context model for tabular regression."""
    def __init__(self, n_estimators=8, **kw):
        super().__init__(**kw)
        self.n_estimators = n_estimators
    def _load(self):
        """Create and fit the TabICL regressor."""
        from tabicl import TabICLRegressor
        self.m = TabICLRegressor(n_estimators=self.n_estimators, kv_cache=True, device=DEVICE, random_state=self.seed,
                                 checkpoint_version="tabicl-regressor-v2-20260212.ckpt")
        self.m.fit(self._frame(self.X_), self.y_)
    def _predict(self, X):
        """Return predictions."""
        return self.m.predict(self._frame(X))
        
class KumoTabularModel(InContextModel):
    """Kumo-Tabular, an in-context model for tabular classification."""
    def __init__(self, size="small", **kw):
        super().__init__(**kw)
        self.size = size
    def _table(self, X, y=None):
        """Convert arrays to an sdm TableTensor, with an optional target column."""
        import sdm
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        stypes = {c: "numerical" for c in df.columns}
        if y is not None:
            df["target"], stypes["target"] = y, "numerical"
        return sdm.TableTensor.from_pandas(df=df, stypes=stypes, device=torch.device(DEVICE))
    def _load(self):
        """Create the model and build the context tables."""
        import sdm
        self.m = sdm.models.KumoTabular(task="regression", size=self.size, device=DEVICE)
        ctx = self._table(self.X_, self.y_)
        self.xc, self.yc = ctx.drop_columns("target"), ctx[:, "target"]
    def _predict(self, X):
        """Return the mean over the predicted quantiles."""
        dev = torch.device(DEVICE)
        with torch.amp.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            out = self.m(x_context=self.xc, y_context=self.yc, x_query=self._table(X), num_estimators=16)
        return out.to_pandas().to_numpy(dtype=np.float64).mean(axis=1)
        
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
    """Fit *model* and return predictions for *X_eval*."""
    p = np.asarray(model.fit(_view(model, X_train), y_train).predict(_view(model, X_eval)), dtype=np.float64)
    if hasattr(model, "release"):
        model.release()
    return p
    
def run_model(name, model, models, P, X_train, y_train, X_eval, raise_errors=False):
    """Fit *model*, storing it in *models* and its predictions in *P*."""
    try:
        P[name] = fit_predict(model, X_train, y_train, X_eval)
        models[name] = model
    except Exception as e:
        if raise_errors:
            raise
        print(f"[{name}] skipped: {type(e).__name__}: {e}")
        
def report(y_true, p):
    """Return RMSE, MAE, median absolute error, R2 and MAPE."""
    return {"RMSE": mean_squared_error(y_true, p) ** 0.5,
            "MAE": mean_absolute_error(y_true, p),
            "MedAE": median_absolute_error(y_true, p),
            "R2": r2_score(y_true, p),
            "MAPE %": 100 * np.mean(np.abs(y_true - p) / np.abs(y_true))}
    
def results_table(P, y_true):
    """Return the metrics table sorted by RMSE."""
    return pd.DataFrame({n: report(y_true, p) for n, p in P.items()}).T.sort_values("RMSE").round(4)
    
def refit_predict(models, best, X, y, X_new):
    """Refit the selected model on (X, y) and predict *X_new*."""
    return fit_predict(models[best], X, y, X_new)