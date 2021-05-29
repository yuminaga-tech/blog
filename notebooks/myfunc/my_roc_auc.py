# -*- coding: utf-8 -*-
import numpy as np
from itertools import product

def my_roc_auc(true_label: np.array, pred_proba : np.array) -> float:
    """
    ROCAUC = Pr (true_labelが0のサンプルの予測スコア < true_labelが1のサンプルの予測スコア) --- (a)
            + 0.5 * Pr (true_labelが0のサンプルの予測スコア = true_labelが1のサンプルの予測スコア) --- (b)
    from sklearn.metrcis import roc_auc_scoreの結果と一致することを確認済み
    """
    # サンプルを抽出
    pred_proba_true_label_0 = pred_proba[true_label == 0]
    pred_proba_true_label_1 = pred_proba[true_label == 1]

    # 全組み合わせを直積から作成
    pred_proba_product = product(pred_proba_true_label_0, pred_proba_true_label_1)
    pred_proba_product = np.array(list(pred_proba_product))

    # 正しい順序で分類できているフラグ (a)
    success_classified_flg = pred_proba_product[:,0] < pred_proba_product[:,1]

    # true_labelが0, 1のサンプル間で予測確率が一致しているフラグ (b)
    equal_score_flg = pred_proba_product[:,0] == pred_proba_product[:,1]

    # どちらもintにして, 足す
    roc_auc_array = success_classified_flg.astype(int) + equal_score_flg.astype(int) * 0.5
    roc_auc = np.mean(roc_auc_array)
    return roc_auc