-- =====================================================================
-- feat_inst: repayment behaviour on previous Home Credit loans.
--
-- Grain trap: one installment can be paid in several partial payments,
-- so installments_payments has several rows per
-- (SK_ID_PREV, NUM_INSTALMENT_VERSION, NUM_INSTALMENT_NUMBER).
-- Step 1 collapses to the true installment grain (sum of payments,
-- date of the LAST payment), step 2 aggregates to the applicant.
-- Without step 1, an installment paid in 3 parts would count 3 times and
-- every partial payment would look like an underpayment.
-- =====================================================================
CREATE OR REPLACE TABLE feat_inst AS
WITH per_installment AS (
    SELECT
        SK_ID_CURR, SK_ID_PREV, NUM_INSTALMENT_VERSION, NUM_INSTALMENT_NUMBER,
        MAX(DAYS_INSTALMENT)     AS days_due,
        MAX(AMT_INSTALMENT)      AS amt_due,
        SUM(AMT_PAYMENT)         AS amt_paid,
        MAX(DAYS_ENTRY_PAYMENT)  AS days_paid          -- NULL = never paid
    FROM raw_installments_payments
    GROUP BY ALL
),
scored AS (
    SELECT *,
        GREATEST(days_paid - days_due, 0)                       AS dpd,     -- days past due
        GREATEST(days_due - days_paid, 0)                       AS dbd,     -- days before due
        (amt_paid < amt_due - 1)::INTEGER                       AS underpaid
    FROM per_installment
)
SELECT
    SK_ID_CURR,
    COUNT(*)                                                    AS INST_N,
    COUNT(DISTINCT SK_ID_PREV)                                  AS INST_N_LOANS,
    AVG((dpd > 0)::INTEGER)                                     AS INST_LATE_RATE,
    AVG((dpd > 30)::INTEGER)                                    AS INST_LATE30_RATE,
    AVG(dpd)                                                    AS INST_AVG_DPD,
    MAX(dpd)                                                    AS INST_MAX_DPD,
    AVG(dbd)                                                    AS INST_AVG_DAYS_EARLY,
    AVG(underpaid)                                              AS INST_UNDERPAID_RATE,
    SUM(amt_paid) / NULLIF(SUM(amt_due), 0)                     AS INST_PAID_TO_DUE_RATIO,
    -- recent behaviour weighs more in practice; keep a last-12-months view
    AVG((dpd > 0)::INTEGER) FILTER (WHERE days_due >= -365)     AS INST_LATE_RATE_LAST_1Y,
    AVG(underpaid)          FILTER (WHERE days_due >= -365)     AS INST_UNDERPAID_RATE_LAST_1Y
FROM scored
GROUP BY SK_ID_CURR;
