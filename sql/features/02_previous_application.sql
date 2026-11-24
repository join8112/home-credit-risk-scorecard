-- =====================================================================
-- feat_prev: the applicant's previous applications at Home Credit itself.
-- Grain in: one row per SK_ID_PREV.  Grain out: one row per SK_ID_CURR.
-- 365243 is again a "not applicable" sentinel in the DAYS_* columns.
-- =====================================================================
CREATE OR REPLACE TABLE feat_prev AS
WITH prev AS (
    SELECT
        * REPLACE (
            NULLIF(DAYS_FIRST_DUE, 365243)   AS DAYS_FIRST_DUE,
            NULLIF(DAYS_LAST_DUE, 365243)    AS DAYS_LAST_DUE,
            NULLIF(DAYS_TERMINATION, 365243) AS DAYS_TERMINATION
        ),
        ROW_NUMBER() OVER (PARTITION BY SK_ID_CURR ORDER BY DAYS_DECISION DESC) AS rn_latest
    FROM raw_previous_application
    WHERE FLAG_LAST_APPL_PER_CONTRACT = 'Y'      -- drop duplicate / superseded records
)
SELECT
    SK_ID_CURR,
    COUNT(*)                                                         AS PREV_N_APPS,
    COUNT(*) FILTER (WHERE NAME_CONTRACT_STATUS = 'Approved')        AS PREV_N_APPROVED,
    COUNT(*) FILTER (WHERE NAME_CONTRACT_STATUS = 'Refused')         AS PREV_N_REFUSED,
    AVG(CASE WHEN NAME_CONTRACT_STATUS = 'Refused' THEN 1.0 ELSE 0.0 END)
                                                                     AS PREV_REFUSED_RATE,
    COUNT(*) FILTER (WHERE DAYS_DECISION >= -365)                    AS PREV_N_APPS_LAST_1Y,
    -MAX(DAYS_DECISION)                                              AS PREV_DAYS_SINCE_LAST_DECISION,
    MAX(CASE WHEN rn_latest = 1 AND NAME_CONTRACT_STATUS = 'Refused' THEN 1 ELSE 0 END)
                                                                     AS PREV_LAST_WAS_REFUSED,
    -- asked vs granted: < 1 means Home Credit gave less than requested
    AVG(AMT_CREDIT / NULLIF(AMT_APPLICATION, 0))
        FILTER (WHERE NAME_CONTRACT_STATUS = 'Approved')             AS PREV_AVG_GRANTED_TO_ASKED,
    AVG(AMT_ANNUITY) FILTER (WHERE NAME_CONTRACT_STATUS = 'Approved') AS PREV_AVG_ANNUITY,
    AVG(CNT_PAYMENT) FILTER (WHERE NAME_CONTRACT_STATUS = 'Approved') AS PREV_AVG_TERM,
    AVG(RATE_DOWN_PAYMENT)                                           AS PREV_AVG_DOWN_PAYMENT_RATE,
    COUNT(*) FILTER (WHERE NAME_YIELD_GROUP = 'high')                AS PREV_N_HIGH_YIELD,
    COUNT(*) FILTER (WHERE NAME_CONTRACT_STATUS = 'Approved'
                       AND DAYS_LAST_DUE IS NULL
                       AND DAYS_TERMINATION IS NULL)                 AS PREV_N_ACTIVE
FROM prev
GROUP BY SK_ID_CURR;
