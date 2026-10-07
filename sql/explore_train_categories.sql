SELECT
    category,
    COUNT(*) AS transactions,
    SUM(isFraud) AS fraud_transactions,
    ROUND(
        100.0 * SUM(isFraud) / COUNT(*), 3
    ) AS fraud_percentage
FROM transactions
WHERE split = 'train'
GROUP BY category
ORDER BY fraud_percentage DESC, transactions DESC;
