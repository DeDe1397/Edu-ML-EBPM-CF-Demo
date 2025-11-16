import numpy as np

def rmse(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.sqrt(((y_true - y_pred)**2).mean()))

def r2(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    ss_res = ((y_true - y_pred)**2).sum()
    ss_tot = ((y_true - y_true.mean())**2).sum()
    return float(1 - ss_res/ss_tot) if ss_tot>0 else 0.0

def precision_at_k(true_items:set, rec_items:list, k:int):
    topk = [i for i,_ in rec_items[:k]]
    return len(set(topk)&set(true_items)) / max(k,1)
