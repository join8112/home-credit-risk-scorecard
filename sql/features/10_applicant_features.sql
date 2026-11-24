-- =====================================================================
-- applicant_features: the modelling table. One row per SK_ID_CURR.
-- LEFT JOINs everywhere: an applicant with no bureau record is information
-- ("thin file"), not a row to drop. HAS_* flags make that explicit, so a
-- scorecard can give "no history" its own bin instead of imputing.
-- =====================================================================
CREATE OR REPLACE TABLE applicant_features AS
SELECT
    a.*,
    -- affordability ratios (what an underwriter looks at first)
    a.AMT_CREDIT  / NULLIF(a.AMT_INCOME_TOTAL, 0)                  AS CREDIT_TO_INCOME,
    a.AMT_ANNUITY / NULLIF(a.AMT_INCOME_TOTAL, 0)                  AS ANNUITY_TO_INCOME,
    a.AMT_CREDIT  / NULLIF(a.AMT_ANNUITY, 0)                       AS CREDIT_TERM_MONTHS,
    a.AMT_CREDIT  / NULLIF(a.AMT_GOODS_PRICE, 0)                   AS CREDIT_TO_GOODS,
    a.AMT_INCOME_TOTAL / NULLIF(a.CNT_FAM_MEMBERS, 0)              AS INCOME_PER_PERSON,
    a.EMPLOYED_YEARS / NULLIF(a.AGE_YEARS, 0)                      AS EMPLOYED_TO_AGE,
    -- external scores: average of whatever is available
    (COALESCE(a.EXT_SOURCE_1, 0) + COALESCE(a.EXT_SOURCE_2, 0) + COALESCE(a.EXT_SOURCE_3, 0))
      / NULLIF((a.EXT_SOURCE_1 IS NOT NULL)::INT + (a.EXT_SOURCE_2 IS NOT NULL)::INT
             + (a.EXT_SOURCE_3 IS NOT NULL)::INT, 0)               AS EXT_SOURCE_MEAN,
    -- history-availability flags
    (b.SK_ID_CURR IS NOT NULL)::INTEGER                            AS HAS_BUREAU,
    (p.SK_ID_CURR IS NOT NULL)::INTEGER                            AS HAS_PREV_APP,
    (i.SK_ID_CURR IS NOT NULL)::INTEGER                            AS HAS_INSTALLMENTS,
    (c.SK_ID_CURR IS NOT NULL)::INTEGER                            AS HAS_CREDIT_CARD,
    -- bureau debt relative to income
    b.BUREAU_ACTIVE_DEBT_SUM / NULLIF(a.AMT_INCOME_TOTAL, 0)       AS BUREAU_DEBT_TO_INCOME,
    b.* EXCLUDE (SK_ID_CURR),
    p.* EXCLUDE (SK_ID_CURR),
    i.* EXCLUDE (SK_ID_CURR),
    s.* EXCLUDE (SK_ID_CURR),
    c.* EXCLUDE (SK_ID_CURR)
FROM stg_application AS a
LEFT JOIN feat_bureau AS b USING (SK_ID_CURR)
LEFT JOIN feat_prev   AS p USING (SK_ID_CURR)
LEFT JOIN feat_inst   AS i USING (SK_ID_CURR)
LEFT JOIN feat_pos    AS s USING (SK_ID_CURR)
LEFT JOIN feat_cc     AS c USING (SK_ID_CURR);
