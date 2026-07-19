"""Real tabular dataset preprocessing for the SGShift reproduction.

Each dataset is reduced to one numeric column per original feature (categoricals
ordinal-encoded), standardized; a binary domain mask encodes the paper's natural
domain split. Cached to <out>.npz. Shapes/splits are printed from real data.

Primary dataset: Diabetes 130-US 30-day readmission (fetch_openml 'Diabetes130US'),
split by Emergency-Room admission (admission_source_id==7) vs non-ER, matching the
paper's ER vs non-ER split. Additional real sets: SUPPORT2 and Adult.
"""
import numpy as np, sys
from sklearn.datasets import fetch_openml

def _ordinal(col):
    s = col.astype('category')
    return s.cat.codes.to_numpy().astype(np.float64)

def _standardize(X):
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0
    return (X - mu) / sd

def prep_diabetes():
    d = fetch_openml('Diabetes130US', version=1, as_frame=True)
    df = d.data.copy(); df['readmit'] = d.target
    # standard preprocessing: drop encounters ending in death/hospice and keep a
    # single (first) encounter per patient, following common practice for this
    # dataset; brings ~101.8k rows toward the paper's 73,615.
    dd = df['discharge_disposition_id'].astype(str)
    df = df[~dd.isin(['11', '13', '14', '19', '20', '21'])]
    df = df.drop_duplicates(subset='patient_nbr', keep='first').reset_index(drop=True)
    y = (df['readmit'].astype(str) == '<30').astype(np.float64).to_numpy()
    dom = (df['admission_source_id'].astype(str) == '7').astype(int).to_numpy()
    drop = ['encounter_id', 'patient_nbr', 'weight', 'payer_code',
            'medical_specialty', 'diag_1', 'diag_2', 'diag_3', 'readmit',
            'admission_source_id']
    feats = [c for c in df.columns if c not in drop]
    cols, names = [], []
    for c in feats:
        col = df[c]
        v = col.to_numpy().astype(np.float64) if col.dtype.kind in 'if' else _ordinal(col)
        if np.nanstd(v) == 0:
            continue
        cols.append(v); names.append(c)
    X = np.column_stack(cols)
    X = np.nan_to_num(X, nan=0.0)
    return X, y, dom, names

def prep_support2():
    d = fetch_openml('support2', version=1, as_frame=True)
    df = d.data.copy()
    tcol = 'hospdead' if 'hospdead' in df.columns else 'death'
    y = df[tcol].astype(str).isin(['1', '1.0', 'True', 'yes']).astype(np.float64).to_numpy()
    age = df['age'].to_numpy().astype(np.float64)
    dom = (age >= 65).astype(int)
    drop = [tcol, 'death', 'hospdead', 'd.time', 'slos', 'charges', 'totcst', 'totmcst']
    feats = [c for c in df.columns if c not in drop]
    cols, names = [], []
    for c in feats:
        col = df[c]
        v = col.to_numpy().astype(np.float64) if col.dtype.kind in 'if' else _ordinal(col)
        if np.nanstd(v) == 0:
            continue
        v = np.nan_to_num(v, nan=0.0)
        cols.append(v); names.append(c)
    X = np.column_stack(cols); X = np.nan_to_num(X, nan=0.0)
    return X, y, dom, names

def prep_adult():
    d = fetch_openml('adult', version=2, as_frame=True)
    df = d.data.copy()
    y = (d.target.astype(str).str.contains('>50K')).astype(np.float64).to_numpy()
    dom = (df['sex'].astype(str).str.strip().str.lower() == 'male').astype(int).to_numpy()
    drop = ['sex', 'fnlwgt', 'education-num']
    feats = [c for c in df.columns if c not in drop]
    cols, names = [], []
    for c in feats:
        col = df[c]
        v = col.to_numpy().astype(np.float64) if col.dtype.kind in 'if' else _ordinal(col)
        if np.nanstd(v) == 0:
            continue
        cols.append(v); names.append(c)
    X = np.column_stack(cols); X = np.nan_to_num(X, nan=0.0)
    return X, y, dom, names

PREP = {'diabetes': prep_diabetes, 'support2': prep_support2, 'adult': prep_adult}

def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return d['X'], d['y'], d['dom'], list(d['names'])

if __name__ == '__main__':
    name, out = sys.argv[1], sys.argv[2]
    X, y, dom, names = PREP[name]()
    Xs = _standardize(X)
    np.savez_compressed(out, X=Xs, y=y, dom=dom, names=np.array(names, dtype=object))
    src = int((dom == 0).sum()); tgt = int((dom == 1).sum())
    print(f"OK {name}: X={X.shape} feats={len(names)} source(non-target)={src} "
          f"target={tgt} pos_rate={y.mean():.4f} "
          f"src_pos={y[dom==0].mean():.4f} tgt_pos={y[dom==1].mean():.4f}")
    print("FEATURES", names)
