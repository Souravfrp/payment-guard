SELECT
    CASE
        WHEN amount < 100 THEN '1. Below 100'
        WHEN amount < 1000 THEN '2. 100 to below 1,000'
        WHEN amount < 10000 THEN '3. 1,000 to below 10,000'
        ELSE '4. 10,000 and above'
    END AS amount_band,
    COUNT(*) AS transactions,
    SUM(isFraud) AS fraud_transactions,
    ROUND(
        100.0 * SUM(isFraud) / COUNT(*), 3
    ) AS fraud_percentage
FROM transactions
WHERE split = 'train'
  AND type = 'TRANSFER'
GROUP BY amount_band
ORDER BY amount_band;