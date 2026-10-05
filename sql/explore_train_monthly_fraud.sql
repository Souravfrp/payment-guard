SELECT
    SUBSTR(timestamp, 1, 7) AS transaction_month,
    COUNT(*) AS transactions,
    SUM(isFraud) AS fraud_transactions,
    ROUND(
        100.0 * SUM(isFraud) / COUNT(*), 3
    ) AS fraud_percentage
FROM transactions
WHERE split = 'train'
GROUP BY transaction_month
ORDER BY transaction_month;