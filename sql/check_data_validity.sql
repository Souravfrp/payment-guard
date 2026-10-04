-- Source references: all three counts should be 1090000.
SELECT
    COUNT(*) AS total_rows,
    COUNT(source_row) AS nonmissing_source_rows,
    COUNT(DISTINCT source_row) AS unique_source_rows
FROM transactions;

-- Each count below should be 0.
SELECT COUNT(*) AS invalid_labels
FROM transactions
WHERE isFraud IS NULL OR isFraud NOT IN (0, 1);

SELECT COUNT(*) AS invalid_amounts
FROM transactions
WHERE amount IS NULL OR amount < 0;

SELECT COUNT(*) AS invalid_splits
FROM transactions
WHERE split IS NULL
   OR split NOT IN ('train', 'validation', 'test');