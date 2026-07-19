"""Authoritative real-dataset loader (gaussianized), caches to _cache/<name>.npz.
diabetes: UCI Diabetes 130-US readmission (OpenML 4541), ER-admission split, 30-day
readmission label. calif: California housing (sklearn), regression, latitude split.
adult: Adult census income (OpenML 'adult'), income>50k, sex split."""
import os, numpy as np
CACHE = os.path.join(os.path.dirname(__file__), '..', '_cache')
os.makedirs(CACHE, exist_ok=True); os.environ.setdefault('SCIKIT_LEARN_DATA', '/tmp/skdata')
AGE_MID = {'[0-10)':5,'[10-20)':15,'[20-30)':25,'[30-40)':35,'[40-50)':45,'[50-60)':55,'[60-70)':65,'[70-80)':75,'[80-90)':85,'[90-100)':95}
DEATH_HOSPICE = {11,13,14,19,20,21}
def _gauss(X):
    from scipy.stats import rankdata, norm
    n = X.shape[0]; Z = np.empty_like(X, float)
    for j in range(X.shape[1]):
        Z[:, j] = norm.ppf((rankdata(X[:, j], method='average') - 0.5) / n)
    return Z
def _encode(feats):
    cols = []
    for c in feats.columns:
        s = feats[c]
        code = s.astype('category').cat.codes.astype(float).values if s.dtype.name in ('object','category') else s.astype(float).values
        if np.nanstd(code) < 1e-8: continue
        v, cnt = np.unique(code[~np.isnan(code)], return_counts=True)
        if cnt.max()/cnt.sum() > 0.999: continue
        cols.append(np.nan_to_num(code, nan=np.nanmean(code)))
    return np.column_stack(cols)
def prep_diabetes():
    from sklearn.datasets import fetch_openml
    df = fetch_openml(data_id=4541, as_frame=True, parser='auto').frame.copy()
    df = df.drop_duplicates(subset='patient_nbr', keep='first')
    df = df[~df['discharge_disposition_id'].astype(int).isin(DEATH_HOSPICE)]
    df = df[df['gender'].isin(['Male','Female'])]
    y = (df['readmitted'].astype(str) == '<30').astype(float).values
    domain = (df['admission_source_id'].astype(int) == 7).astype(int).values
    feats = df.drop(columns=['encounter_id','patient_nbr','weight','payer_code','medical_specialty','diag_1','diag_2','diag_3','readmitted','admission_source_id']).copy()
    feats['age'] = feats['age'].map(AGE_MID).astype(float)
    X = _encode(feats); keep = ~np.isnan(X).any(1)
    return _gauss(X[keep]), domain[keep].astype(int), y[keep], 'clf'
def prep_calif():
    from sklearn.datasets import fetch_california_housing
    d = fetch_california_housing(); X = d.data.astype(float); y = d.target.astype(float)
    domain = (X[:, 6] > np.median(X[:, 6])).astype(int)   # Latitude north/south
    return _gauss(X), domain.astype(int), (y - y.mean()) / y.std(), 'reg'
def prep_adult():
    from sklearn.datasets import fetch_openml
    df = fetch_openml('adult', version=2, as_frame=True, parser='auto').frame.copy()
    y = df['class'].astype(str).str.contains('>50K').astype(float).values
    domain = (df['sex'].astype(str) == 'Female').astype(int).values
    X = _encode(df.drop(columns=['class','sex']).copy())
    return _gauss(X), domain.astype(int), y, 'clf'
PREPS = {'diabetes': prep_diabetes, 'calif': prep_calif, 'adult': prep_adult}
def load(name):
    path = os.path.join(CACHE, f'{name}.npz')
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True); return z['Zc'], z['domain'], z['y'], str(z['task'])
    Zc, domain, y, task = PREPS[name]()
    with open(path, 'wb') as fh: np.savez_compressed(fh, Zc=Zc, domain=domain, y=y, task=task)
    return Zc, domain, y, task
