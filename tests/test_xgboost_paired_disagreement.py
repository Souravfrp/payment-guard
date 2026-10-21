"""Synthetic checks only: tests never touch Payment Guard's real saved data."""
import numpy as np
import pandas as pd
import pytest

from scripts.plot_xgboost_paired_disagreement import (
    BUDGETS, make_chart, paired_row, topk,
)


def toy():
    return pd.DataFrame({"isFraud":[1,1,1,0,0,0,0,0],
                         "tie_priority":[7,0,5,2,3,1,4,6],
                         "logistic":[.9,.8,.1,.4,.3,.2,.0,.01],
                         "xgboost":[.9,.1,.8,.4,.3,.2,.0,.01]})


def test_tie_ranking_uses_label_independent_priority():
    assert topk([.8,.8,.4], [2,0,1], 1).tolist() == [False,True,False]


def test_invalid_priority_rejected():
    with pytest.raises(ValueError,match="permutation"):
        topk([.8,.7,.6], [1,1,1], 2)


def test_budget_uses_ceil_rounding():
    f=toy()
    r=paired_row("January",0.001,f,"logistic","xgboost")
    assert r["selected_each_model"]==1
    assert r["both_caught"] == 1


def test_positive_disagreement_partition_and_difference():
    f=toy()
    r=paired_row("January",0.25,f,"logistic","xgboost")
    assert (r["both_caught"],r["only_logistic"],r["only_xgboost"],r["neither_caught"]) == (1,1,1,0)
    assert r["logistic_caught"]==r["xgboost_caught"]==2
    assert r["xgboost_minus_logistic"]==0


def test_selected_overlaps_include_negative_rows():
    f=toy()
    r=paired_row("February",0.5,f,"logistic","xgboost")
    assert r["selected_each_model"]==4
    assert r["both_selected_all_labels"]==3
    assert r["selected_only_logistic_all_labels"]==r["selected_only_xgboost_all_labels"]==1


def test_rejects_invalid_binary_labels():
    f=toy(); f.loc[0,"isFraud"]=2
    with pytest.raises(ValueError,match="binary"):
        paired_row("January",.25,f,"logistic","xgboost")


def test_empty_review_budget_rejected():
    with pytest.raises(ValueError,match="outside population"):
        topk([0.1,0.2], [0,1],0)


def test_figure_is_generated_only_from_aggregated_rows(tmp_path):
    rows=[]
    for month,p in (("January",281),("February",202)):
        for f in BUDGETS:
            rows.append({"month":month,"review_fraction":f,"review_budget_percent":f*100,
                         "fraud_labels":p,"selected_each_model":int(np.ceil(f*180000)),
                         "both_caught":p-10,"only_logistic":2,"only_xgboost":3,
                         "neither_caught":5,"logistic_caught":p-8,"xgboost_caught":p-7})
    path=tmp_path/"paired.png"
    make_chart(pd.DataFrame(rows),path)
    assert path.exists() and path.stat().st_size>10000
