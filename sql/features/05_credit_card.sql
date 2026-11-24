-- =====================================================================
-- feat_cc: monthly snapshots of previous Home Credit credit cards.
-- Utilisation (balance / limit) is one of the classic scorecard variables.
-- =====================================================================
CREATE OR REPLACE TABLE feat_cc AS
WITH m AS (
    SELECT *,
        AMT_BALANCE / NULLIF(AMT_CREDIT_LIMIT_ACTUAL, 0)            AS utilisation,
        AMT_PAYMENT_TOTAL_CURRENT / NULLIF(AMT_INST_MIN_REGULARITY, 0) AS pay_to_min
    FROM raw_credit_card_balance
)
SELECT
    SK_ID_CURR,
    COUNT(DISTINCT SK_ID_PREV)                                  AS CC_N_CARDS,
    COUNT(*)                                                    AS CC_N_MONTHS,
    AVG(utilisation)                                            AS CC_AVG_UTILISATION,
    MAX(utilisation)                                            AS CC_MAX_UTILISATION,
    AVG(utilisation) FILTER (WHERE MONTHS_BALANCE >= -6)        AS CC_AVG_UTILISATION_LAST_6M,
    AVG(AMT_DRAWINGS_ATM_CURRENT)                               AS CC_AVG_ATM_DRAWINGS,
    AVG(LEAST(pay_to_min, 10))                                  AS CC_AVG_PAY_TO_MIN,   -- capped
    MAX(SK_DPD_DEF)                                             AS CC_MAX_DPD_DEF,
    AVG((SK_DPD_DEF > 0)::INTEGER)                              AS CC_SHARE_MONTHS_DPD
FROM m
GROUP BY SK_ID_CURR;
