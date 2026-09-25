"""Shared data code for my Lane 2 (refresh / decline) notebooks, weeks 3-5.

One place builds the page-level frame, the baseline rule and the split, so the
baseline (w04) and the model (w05) use exactly the same rows, labels and test clients.

Decision moment: end of 2026-03-15.
Features use 1-15 March only; the label uses 16-31 March only.
"""
import os

import duckdb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

REL = "hf://datasets/FlyRank/internship-warehouse"
MONTH = "2026-03"
DECISION_DATE = "2026-03-15"
SEED = 42

FACT = f"read_parquet('{REL}/fact_content_daily_performance/month={MONTH}/*.parquet')"
DIM_CONTENT = f"read_parquet('{REL}/dim_content.parquet')"


def connect():
    """DuckDB connection with the Hugging Face token from the env or the local login."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        from huggingface_hub import get_token
        token = get_token()
    if not token:
        import getpass
        token = getpass.getpass("Hugging Face READ token: ")
    con = duckdb.connect()
    con.execute(f"CREATE OR REPLACE SECRET hf (TYPE huggingface, TOKEN '{token}')")
    return con


def build_frame(con):
    """One row per page: first-half (1-15 Mar) signals + second-half label."""
    frame = con.sql(f"""
        WITH d AS (
            SELECT client_hash_id, content_hash_id, report_date,
                   gsc_impressions, gsc_clicks, gsc_sum_position
            FROM {FACT}
            WHERE gsc_data_available IS TRUE
        ),
        agg AS (
            SELECT client_hash_id, content_hash_id,
                SUM(gsc_impressions)  FILTER (WHERE report_date <= DATE '{DECISION_DATE}') AS imp_h1,
                SUM(gsc_clicks)       FILTER (WHERE report_date <= DATE '{DECISION_DATE}') AS clk_h1,
                SUM(gsc_sum_position) FILTER (WHERE report_date <= DATE '{DECISION_DATE}') AS sumpos_h1,
                COUNT(*) FILTER (WHERE report_date <= DATE '{DECISION_DATE}' AND gsc_impressions > 0) AS active_days_h1,
                SUM(gsc_impressions)  FILTER (WHERE report_date <= DATE '2026-03-07') AS imp_wk1,
                SUM(gsc_impressions)  FILTER (WHERE report_date >  DATE '2026-03-07'
                                                AND report_date <= DATE '{DECISION_DATE}') AS imp_wk2,
                SUM(gsc_impressions)  FILTER (WHERE report_date >  DATE '{DECISION_DATE}') AS imp_h2
            FROM d
            GROUP BY 1, 2
            HAVING imp_h1 >= 50
        )
        SELECT a.*, c.content_created_date, c.content_type
        FROM agg a
        LEFT JOIN {DIM_CONTENT} c USING (client_hash_id, content_hash_id)
    """).df().fillna({"imp_h2": 0, "imp_wk1": 0, "imp_wk2": 0})

    decision = pd.Timestamp(DECISION_DATE)
    frame["log_imp_h1"] = np.log1p(frame["imp_h1"])
    frame["ctr_h1"] = frame["clk_h1"] / frame["imp_h1"]
    # gsc_sum_position = 0 means no position was recorded, not rank 0
    frame["avg_pos_h1"] = np.where(frame["sumpos_h1"] > 0, frame["sumpos_h1"] / frame["imp_h1"], np.nan)
    frame["momentum_h1"] = (frame["imp_wk2"] / 8) / ((frame["imp_wk1"] / 7) + 1)
    frame["age_days"] = (decision - pd.to_datetime(frame["content_created_date"])).dt.days

    # Expected CTR = median CTR of pages in the same position bucket (first-half data only)
    frame["pos_bucket"] = pd.cut(frame["avg_pos_h1"], [0, 3, 5, 10, 20, 1000],
                                 labels=["1-3", "3-5", "5-10", "10-20", "20+"])
    expected = frame.groupby("pos_bucket", observed=True)["ctr_h1"].transform("median")
    frame["ctr_gap"] = frame["ctr_h1"] - expected

    # Label: average daily impressions in 16-31 Mar (16 days) below 80% of 1-15 Mar (15 days)
    frame["is_declining"] = ((frame["imp_h2"] / 16) < 0.8 * (frame["imp_h1"] / 15)).astype(int)
    return frame.drop(columns=["imp_h2"])   # the label stays, the future column goes


def baseline_score(frame):
    """The frozen Week-4 rule: a score, ONE reason code and an action label per page."""
    on_page_one = frame["avg_pos_h1"] <= 10
    ctr_under = on_page_one & (frame["ctr_gap"] < 0)
    settling_age = frame["age_days"].between(30, 180)

    points = 2 * ctr_under.astype(int) + 1 * settling_age.astype(int)
    out = frame.copy()
    out["baseline_score"] = points * frame["log_imp_h1"]   # ties broken by traffic at risk
    out["reason_code"] = np.select(
        [ctr_under & settling_age, ctr_under, settling_age],
        ["ctr_below_position_young_page", "ctr_below_position", "young_page_settling"],
        "no_flag")
    out["action"] = np.select(
        [ctr_under, settling_age],
        ["rewrite_title_and_snippet", "monitor"],
        "no_action")
    return out


def client_split(frame):
    """Same 75/25 client-grouped split in every notebook (seed 42)."""
    split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_idx, test_idx = next(split.split(frame, frame["is_declining"], groups=frame["client_hash_id"]))
    return frame.iloc[train_idx].copy(), frame.iloc[test_idx].copy()


def precision_at_k(scores, labels, k):
    order = np.argsort(-np.asarray(scores), kind="stable")
    return float(np.asarray(labels)[order[:k]].mean())
