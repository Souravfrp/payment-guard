SELECT COUNT(*) AS missing_timestamps
FROM transactions
WHERE timestamp IS NULL
   OR TRIM(timestamp) = '';
