TASK = {
    "id": 3,
    "name": "Hard - Top Customers Revenue Optimization",
    "description": "Find the top 5 customers by total spending in the last 12 months. The query has both a logic bug and performance bottlenecks that should be improved with query rewrite plus index usage.",
    "difficulty": "hard",
    "broken_query": """
SELECT
    c.id,
    c.name,
    SUM(oi.quantity * p.price) AS total_spent
FROM customers c
JOIN orders o ON o.customer_id = c.id
JOIN order_items oi ON oi.order_id = o.id
JOIN products p ON p.id = oi.product_id
WHERE o.order_date >= date('now', '-12 months')
GROUP BY c.id, c.name
ORDER BY total_spent ASC
LIMIT 5;
""".strip(),
    "expected_query": """
SELECT
    c.id,
    c.name,
    ROUND(SUM(oi.quantity * p.price), 2) AS total_spent
FROM orders o
JOIN customers c ON c.id = o.customer_id
JOIN order_items oi ON oi.order_id = o.id
JOIN products p ON p.id = oi.product_id
WHERE o.order_date >= date('now', '-12 months')
GROUP BY c.id, c.name
ORDER BY total_spent DESC
LIMIT 5;
""".strip(),
    "expected_rows": [],
    "baseline_cost": 0,
}
