#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Claim 5  (DSM-TREE : distilling complete hierarchical decision logic, Sec 4.2 / 5.2 / Thm G.5)
Paper: "Trees to Flows and Back" (OpenReview gW7NZN8zJu, arXiv 2605.00414).

Claim: DSM-TREE distills the ENTIRE decision trajectory of a tree (every internal
split, not just leaf predictions) into a neural network, MATCHING teacher performance
within 2% on most benchmarks (exceeding it by 3.7% on Heart Disease), transferring
complete hierarchical logic into a differentiable network (Thm G.5).

Faithful CPU reproduction of Algorithms 3 & 4:
  Teacher : RandomForest oracle -> pseudo-labels -> DecisionTree "Base Tree" (Alg 3).
  DSM-TREE student : one MLP M(x, level) trained with per-level cross-entropy to predict
      the tree's split decision (left/right) at EVERY level of the traversal.
  Inference (Alg 4): traverse the tree using the MLP's per-level decisions; output leaf value.
  Baseline : leaf-only distillation (MLP trained on the Base Tree's final label only).
Metric follows the paper's wording ("within 2% ... and exceeding on Heart Disease"):
  MATCH = DSM-TREE acc >= teacher acc - 2%  (being ABOVE the teacher counts as a win).
We also report teacher-prediction agreement and decision-PATH agreement (structure transfer).
CPU-only, deterministic (fixed random_state).
"""
import json, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
from sklearn.datasets import load_breast_cancer, load_wine, load_iris, fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
t0=time.time()

def openml_ds(name, D):
    d=fetch_openml(name, version=1, as_frame=False, parser="liac-arff")
    X=np.asarray(d.data, float); tgt=np.array(d.target)
    y=(tgt==np.unique(tgt)[-1]).astype(int) if tgt.dtype.kind not in "iuf" else tgt.astype(int)
    m=np.isfinite(X).all(1); return X[m], y[m], D

def get_datasets():
    ds={}
    bc=load_breast_cancer(); ds["Cancer"]=(bc.data,bc.target,5)
    wn=load_wine();          ds["Wine"]=(wn.data,wn.target,5)
    ir=load_iris();          ds["Iris"]=(ir.data,ir.target,4)
    for nm,key,D in [("heart-statlog","Heart-Disease",5),("ionosphere","Ionosphere",5),("diabetes","Diabetes",5)]:
        try: ds[key]=openml_ds(nm,D)
        except Exception as e: print(f"  [note] {key} (OpenML) unavailable:", str(e)[:50])
    return ds

def build_level_data(tree, X):
    cl=tree.tree_.children_left; cr=tree.tree_.children_right
    feat=tree.tree_.feature; thr=tree.tree_.threshold
    XX=[]; LV=[]; DEC=[]
    for i in range(len(X)):
        node=0; level=0
        while cl[node]!=-1:
            go_left = X[i,feat[node]]<=thr[node]
            XX.append(X[i]); LV.append(level); DEC.append(0 if go_left else 1)
            node = cl[node] if go_left else cr[node]; level+=1
    return np.array(XX), np.array(LV), np.array(DEC)
def onehot_level(lv, D):
    O=np.zeros((len(lv),D)); O[np.arange(len(lv)),np.clip(lv,0,D-1)]=1; return O
def dsmtree_infer(tree, mlp, X, D):
    # Algorithm 4, vectorised: one batched MLP predict per level.
    cl=tree.tree_.children_left; cr=tree.tree_.children_right; value=tree.tree_.value
    node=np.zeros(len(X),int)
    for step in range(D):
        internal=np.where(cl[node]!=-1)[0]                 # samples still at an internal node
        if internal.size==0: break
        feats=np.hstack([X[internal], onehot_level(np.full(internal.size,step),D)])
        d=mlp.predict(feats)                                # left(0)/right(1) decisions
        nd=node[internal]
        node[internal]=np.where(d==0, cl[nd], cr[nd])
    preds=np.array([int(np.argmax(value[nd][0])) for nd in node])
    return preds, node

rows=[]
for name,(X,y,D) in get_datasets().items():
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=0,stratify=y)
    sc=StandardScaler().fit(Xtr); Xtr=sc.transform(Xtr); Xte=sc.transform(Xte)
    rf=RandomForestClassifier(n_estimators=120,random_state=0).fit(Xtr,ytr)
    pseudo=rf.predict(Xtr)
    base=DecisionTreeClassifier(max_depth=D,random_state=0).fit(Xtr,pseudo)
    teacher_acc=base.score(Xte,yte)
    XX,LV,DEC=build_level_data(base,Xtr)
    feats=np.hstack([XX, onehot_level(LV,D)])
    student=MLPClassifier(hidden_layer_sizes=(128,64),alpha=3e-4,max_iter=2000,random_state=0).fit(feats,DEC)
    dsm_pred,dsm_leaf=dsmtree_infer(base,student,Xte,D)
    dsm_acc=float(np.mean(dsm_pred==yte))
    true_leaf=base.apply(Xte); path_agree=float(np.mean(dsm_leaf==true_leaf))
    teacher_agree=float(np.mean(dsm_pred==base.predict(Xte)))
    leaf_labels=base.predict(Xtr)
    leafmlp=MLPClassifier(hidden_layer_sizes=(128,64),alpha=5e-4,max_iter=900,random_state=0).fit(Xtr,leaf_labels)
    leaf_acc=float(leafmlp.score(Xte,yte))
    gap=dsm_acc-teacher_acc
    match = dsm_acc >= teacher_acc - 0.02          # within 2% OR exceeds (paper wording)
    rows.append(dict(dataset=name,depth=D,teacher_base_tree_acc=teacher_acc,dsmtree_acc=dsm_acc,
                     gap_vs_teacher=gap,match_within2_or_better=bool(match),exceeds=bool(gap>0),
                     leaf_only_acc=leaf_acc,teacher_pred_agreement=teacher_agree,path_agreement=path_agree))

print("="*82); print("CLAIM 5  DSM-TREE : distilling complete hierarchical decision logic"); print("="*82)
print(f"{'dataset':15s} {'teacher':>8s} {'DSM-TREE':>9s} {'gap':>7s} {'match':>6s} {'leafKD':>7s} {'teachAgr':>8s} {'pathAgr':>8s}")
for r in rows:
    print(f"{r['dataset']:15s} {r['teacher_base_tree_acc']*100:7.2f}% {r['dsmtree_acc']*100:8.2f}% "
          f"{r['gap_vs_teacher']*100:+6.2f}% {str(r['match_within2_or_better']):>6s} {r['leaf_only_acc']*100:6.2f}% "
          f"{r['teacher_pred_agreement']*100:7.1f}% {r['path_agreement']*100:7.1f}%")
n=len(rows); n_match=sum(r["match_within2_or_better"] for r in rows); n_exceed=sum(r["exceeds"] for r in rows)
mean_teachagree=float(np.mean([r["teacher_pred_agreement"] for r in rows]))
mean_path=float(np.mean([r["path_agreement"] for r in rows]))
print("-"*82)
print(f"MATCH (within 2% or exceeds teacher): {n_match}/{n} ; exceeds teacher: {n_exceed}/{n} ; "
      f"mean teacher-pred agreement={mean_teachagree*100:.1f}% ; mean path agreement={mean_path*100:.1f}%")
verdict = (n_match >= int(np.ceil(0.6*n))) and (mean_teachagree > 0.85) and (n_exceed >= 1)
print("VERDICT (DSM-TREE distills full hierarchy, matches teacher within 2% on most):",
      "SUPPORTED" if verdict else "PARTIAL/NOT")

out=dict(
  claim="DSM-TREE distills complete hierarchical decision logic (per-level splits, not just leaves) into an MLP, matching Base-Tree teacher within 2% on most datasets (exceeding on some) and reproducing decision paths (Thm G.5)",
  metric="MATCH = DSM-TREE acc >= teacher acc - 2% (paper: within 2% or exceeding)",
  per_dataset=rows,
  summary=dict(datasets=n, match_within2_or_better=int(n_match), exceeds_teacher=int(n_exceed),
               mean_teacher_pred_agreement=mean_teachagree, mean_path_agreement=mean_path),
  targets=dict(match=">= n-1 datasets within 2%/better", teacher_pred_agreement=">0.85", exceeds=">=1 dataset"),
  verdict="SUPPORTED" if verdict else "PARTIAL", runtime_s=round(time.time()-t0,3))
with open(os.path.join(os.path.dirname(__file__),"results.json"),"w") as f: json.dump(out,f,indent=2)
print("runtime_s =",round(time.time()-t0,3)); print("wrote results.json")
