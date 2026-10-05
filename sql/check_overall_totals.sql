SELECT
    COUNT(*) AS total_transactions,
    SUM(isFraud) AS positive_labels,
    ROUND(100.0 * SUM(isFraud) / COUNT(*), 5) AS positive_percentage
FROM transactions;