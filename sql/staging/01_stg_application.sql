-- =====================================================================
-- stg_application: train + test applications in one table, cleaned.
-- Grain: one row per SK_ID_CURR (one loan application).
--
-- Cleaning decisions (each one is discussed in reports/01_data_and_eda.md):
--   * DAYS_EMPLOYED = 365243 is a sentinel (~1,000 years) used for
--     pensioners / unemployed. Replace with NULL and keep an explicit flag,
--     because the flag itself carries risk information.
--   * CODE_GENDER = 'XNA' (4 rows) -> NULL.
--   * All DAYS_* columns are negative offsets from the application date;
--     convenience columns in years are added with positive sign.
-- =====================================================================
CREATE OR REPLACE TABLE stg_application AS
WITH unioned AS (
    SELECT *, TRUE  AS IS_TRAIN FROM raw_application_train
    UNION ALL BY NAME
    SELECT *, FALSE AS IS_TRAIN FROM raw_application_test      -- TARGET is NULL here
)
SELECT
    * REPLACE (
        CASE WHEN DAYS_EMPLOYED = 365243 THEN NULL ELSE DAYS_EMPLOYED END AS DAYS_EMPLOYED,
        NULLIF(CODE_GENDER, 'XNA')                                        AS CODE_GENDER,
        NULLIF(ORGANIZATION_TYPE, 'XNA')                                  AS ORGANIZATION_TYPE
    ),
    (DAYS_EMPLOYED = 365243)::INTEGER                                     AS FLAG_EMPLOYED_ANOMALY,
    -DAYS_BIRTH / 365.25                                                  AS AGE_YEARS,
    CASE WHEN DAYS_EMPLOYED = 365243 THEN NULL
         ELSE -DAYS_EMPLOYED / 365.25 END                                 AS EMPLOYED_YEARS
FROM unioned;
