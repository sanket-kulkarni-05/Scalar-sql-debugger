TASK = {
    "id": 1,
    "name": "Easy - Customer Orders Filter Fix",
    "description": "Fix a simple correctness bug: return customer names for orders above $250, but the current query uses the wrong comparison direction.",
    "difficulty": "easy",
    "broken_query": """
SELECT c.name, o.total_amount
FROM customers c
JOIN orders o ON c.id = o.customer_id
WHERE o.total_amount < 250
ORDER BY o.total_amount DESC
LIMIT 20;
""".strip(),
    "expected_query": """
SELECT c.name, o.total_amount
FROM customers c
JOIN orders o ON c.id = o.customer_id
WHERE o.total_amount > 250
ORDER BY o.total_amount DESC
LIMIT 20;
""".strip(),
    "expected_rows": [],
    "baseline_cost": 0,
}
