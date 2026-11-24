-- =====================================================================
-- feat_pos: monthly snapshots of previous POS (point-of-sale) and cash loans.
-- Grain in: SK_ID_PREV x MONTHS_BALANCE.  Grain out: SK_ID_CURR.
-- =====================================================================
CREATE OR REPLACE TABLE feat_pos AS
WITH latest AS (   -- most recent snapshot of each loan
    SELECT * FROM raw_pos_cash_balance
    QUALIFY ROW_NUMBER() OVER (PARTITION BY SK_ID_PREV ORDER BY MONTHS_BALANCE DESC) = 1
)
SELECT
    p.SK_ID_CURR,
    COUNT(*)                                                    AS POS_N_MONTHS,
    COUNT(DISTINCT p.SK_ID_PREV)                                AS POS_N_LOANS,
    MAX(p.SK_DPD_DEF)                                           AS POS_MAX_DPD_DEF,
    AVG((p.SK_DPD_DEF > 0)::INTEGER)                            AS POS_SHARE_MONTHS_DPD,
    AVG((p.SK_DPD_DEF > 0)::INTEGER) FILTER (WHERE p.MONTHS_BALANCE >= -12)
                                                                AS POS_SHARE_MONTHS_DPD_LAST_12M,
    ANY_VALUE(l.n_active)                                       AS POS_N_ACTIVE,
    ANY_VALUE(l.remaining_instalments)                          AS POS_REMAINING_INSTALMENTS
FROM raw_pos_cash_balance AS p
LEFT JOIN (
    SELECT SK_ID_CURR,
           COUNT(*) FILTER (WHERE NAME_CONTRACT_STATUS = 'Active') AS n_active,
           SUM(CNT_INSTALMENT_FUTURE)
               FILTER (WHERE NAME_CONTRACT_STATUS = 'Active')      AS remaining_instalments
    FROM latest GROUP BY SK_ID_CURR
) AS l USING (SK_ID_CURR)
GROUP BY p.SK_ID_CURR;
