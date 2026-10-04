SELECT
    split,
    COUNT(*) AS total_transactions,
    SUM(isFraud) AS positive_labels,
    MIN(timestamp) AS earliest_transaction,
    MAX(timestamp) AS latest_transaction
FROM transactions
GROUP BY split
ORDER BY MIN(timestamp);