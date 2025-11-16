import numpy as np

def rmse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.sqrt(((y_true - y_pred)**2).mean()))

def r2(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ss_res = ((y_true - y_pred)**2).sum()
    ss_tot = ((y_true - y_true.mean())**2).sum()
    return float(1 - ss_res/ss_tot) if ss_tot>0 else 0.0

def precision_at_k(true_items, recs, k: int) -> float:
    if k <= 0:
        return 0.0
    if not true_items:
        return 0.0
    topk = [it for it, _ in recs[:k]]
    hit = len(set(topk) & set(true_items))
    return hit / k

def recall_at_k(true_items, recs, k: int) -> float:
    if not true_items:
        return 0.0
    if k <= 0:
        return 0.0
    topk = [it for it, _ in recs[:k]]
    hit = len(set(topk) & set(true_items))
    return hit / len(true_items)