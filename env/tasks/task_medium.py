TASK = {
    "id": 2,
    "name": "Medium - Category Sales Aggregation",
    "description": "Fix aggregation semantics: count sold items per category in the last 365 days; the broken query incorrectly groups by product and inflates category rows.",
    "difficulty": "medium",
    "broken_query": """
SELECT cat.name AS category_name, COUNT(oi.id) AS item_lines
FROM categories cat
JOIN products p ON p.category_id = cat.id
JOIN order_items oi ON oi.product_id = p.id
JOIN orders o ON o.id = oi.order_id
WHERE o.order_date >= date('now', '-365 days')
GROUP BY cat.name, p.id
ORDER BY item_lines DESC
LIMIT 10;
""".strip(),
    "expected_query": """
SELECT cat.name AS category_name, SUM(oi.quantity) AS items_sold
FROM categories cat
JOIN products p ON p.category_id = cat.id
JOIN order_items oi ON oi.product_id = p.id
JOIN orders o ON o.id = oi.order_id
WHERE o.order_date >= date('now', '-365 days')
GROUP BY cat.name
ORDER BY items_sold DESC;
""".strip(),
    "expected_rows": [],
    "baseline_cost": 0,
}
