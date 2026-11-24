-- =====================================================================
-- feat_bureau: credit history at OTHER institutions (Credit Bureau).
-- Join path:  bureau_balance --(SK_ID_BUREAU)--> bureau --(SK_ID_CURR)--> applicant
--
-- Two-step aggregation, because the grains differ:
--   1. bureau_balance (monthly status per bureau loan) -> one row per SK_ID_BUREAU
--   2. bureau (+ step 1)                               -> one row per SK_ID_CURR
-- Aggregating bureau_balance straight to SK_ID_CURR would let loans with
-- long histories dominate the averages.
-- STATUS codes: C = closed, X = unknown, 0 = no DPD, 1 = 1-30 DPD, ... 5 = 120+ DPD / written off.
-- =====================================================================
CREATE OR REPLACE TABLE feat_bureau AS
WITH bb_per_loan AS (
    SELECT
        SK_ID_BUREAU,
        COUNT(*)                                                            AS bb_months,
        MAX(CASE WHEN STATUS IN ('1','2','3','4','5') THEN CAST(STATUS AS INTEGER) ELSE 0 END)
                                                                            AS bb_worst_status,
        AVG(CASE WHEN STATUS IN ('1','2','3','4','5') THEN 1.0 ELSE 0.0 END) AS bb_share_months_dpd,
        MAX(CASE WHEN STATUS IN ('1','2','3','4','5') AND MONTHS_BALANCE >= -12
                 THEN 1 ELSE 0 END)                                         AS bb_dpd_last_12m
    FROM raw_bureau_balance
    GROUP BY SK_ID_BUREAU
),
loans AS (
    SELECT b.*, bb.bb_months, bb.bb_worst_status, bb.bb_share_months_dpd, bb.bb_dpd_last_12m
    FROM raw_bureau AS b
    LEFT JOIN bb_per_loan AS bb USING (SK_ID_BUREAU)
)
SELECT
    SK_ID_CURR,
    COUNT(*)                                                        AS BUREAU_N_LOANS,
    COUNT(*) FILTER (WHERE CREDIT_ACTIVE = 'Active')                AS BUREAU_N_ACTIVE,
    COUNT(*) FILTER (WHERE CREDIT_ACTIVE = 'Bad debt')              AS BUREAU_N_BAD_DEBT,
    COUNT(DISTINCT CREDIT_TYPE)                                     AS BUREAU_N_CREDIT_TYPES,
    COUNT(*) FILTER (WHERE CREDIT_TYPE = 'Microloan')               AS BUREAU_N_MICROLOAN,
    COUNT(*) FILTER (WHERE DAYS_CREDIT >= -365)                     AS BUREAU_N_OPENED_LAST_1Y,
    -MAX(DAYS_CREDIT)                                               AS BUREAU_DAYS_SINCE_LAST_OPEN,
    -MIN(DAYS_CREDIT)                                               AS BUREAU_DAYS_SINCE_FIRST_OPEN,
    -- exposure on currently active loans
    SUM(AMT_CREDIT_SUM)      FILTER (WHERE CREDIT_ACTIVE = 'Active') AS BUREAU_ACTIVE_CREDIT_SUM,
    -- a few bureau records carry negative debt (credit balance); floor at 0
    SUM(GREATEST(AMT_CREDIT_SUM_DEBT, 0)) FILTER (WHERE CREDIT_ACTIVE = 'Active') AS BUREAU_ACTIVE_DEBT_SUM,
    SUM(GREATEST(AMT_CREDIT_SUM_DEBT, 0)) FILTER (WHERE CREDIT_ACTIVE = 'Active')
        / NULLIF(SUM(AMT_CREDIT_SUM) FILTER (WHERE CREDIT_ACTIVE = 'Active'), 0)
                                                                    AS BUREAU_ACTIVE_DEBT_RATIO,
    -- delinquency
    SUM(AMT_CREDIT_SUM_OVERDUE)                                     AS BUREAU_OVERDUE_SUM,
    MAX(CREDIT_DAY_OVERDUE)                                         AS BUREAU_MAX_DAYS_OVERDUE,
    MAX(AMT_CREDIT_MAX_OVERDUE)                                     AS BUREAU_MAX_AMT_OVERDUE,
    SUM(CNT_CREDIT_PROLONG)                                         AS BUREAU_N_PROLONG,
    -- from bureau_balance
    MAX(bb_worst_status)                                            AS BUREAU_BB_WORST_STATUS,
    AVG(bb_share_months_dpd)                                        AS BUREAU_BB_AVG_SHARE_DPD,
    MAX(bb_dpd_last_12m)                                            AS BUREAU_BB_DPD_LAST_12M,
    COUNT(bb_months)                                                AS BUREAU_N_LOANS_WITH_BB
FROM loans
GROUP BY SK_ID_CURR;
