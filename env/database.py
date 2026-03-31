from __future__ import annotations

import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from faker import Faker


def seed_database(db_path: str) -> None:
    random.seed(42)
    Faker.seed(42)
    fake = Faker()

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("PRAGMA foreign_keys = ON;")

    cursor.executescript(
        """
        DROP TABLE IF EXISTS order_items;
        DROP TABLE IF EXISTS orders;
        DROP TABLE IF EXISTS products;
        DROP TABLE IF EXISTS categories;
        DROP TABLE IF EXISTS customers;

        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            city TEXT NOT NULL
        );

        CREATE TABLE categories (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE
        );

        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category_id INTEGER NOT NULL,
            price REAL NOT NULL,
            FOREIGN KEY (category_id) REFERENCES categories(id)
        );

        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            order_date TEXT NOT NULL,
            total_amount REAL NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        );

        CREATE TABLE order_items (
            id INTEGER PRIMARY KEY,
            order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        );
        """
    )

    category_names = [
        "Electronics",
        "Home",
        "Books",
        "Apparel",
        "Sports",
        "Beauty",
    ]

    cursor.executemany(
        "INSERT INTO categories (id, name) VALUES (?, ?)",
        [(idx + 1, name) for idx, name in enumerate(category_names)],
    )

    customers = []
    for idx in range(1, 101):
        customers.append(
            (
                idx,
                fake.name(),
                f"customer{idx}@example.com",
                fake.city(),
            )
        )

    cursor.executemany(
        "INSERT INTO customers (id, name, email, city) VALUES (?, ?, ?, ?)",
        customers,
    )

    products = []
    for idx in range(1, 31):
        category_id = random.randint(1, len(category_names))
        price = round(random.uniform(8.0, 300.0), 2)
        products.append((idx, fake.word().capitalize(), category_id, price))

    cursor.executemany(
        "INSERT INTO products (id, name, category_id, price) VALUES (?, ?, ?, ?)",
        products,
    )

    base_date = datetime.utcnow().date()
    orders = []
    order_items = []

    order_item_id = 1
    for order_id in range(1, 301):
        customer_id = random.randint(1, 100)
        days_back = random.randint(0, 730)
        order_date = base_date - timedelta(days=days_back)

        item_count = random.randint(1, 5)
        chosen_products = random.sample(range(1, 31), k=item_count)
        computed_total = 0.0

        for product_id in chosen_products:
            quantity = random.randint(1, 4)
            price = products[product_id - 1][3]
            computed_total += price * quantity
            order_items.append((order_item_id, order_id, product_id, quantity))
            order_item_id += 1

        orders.append((order_id, customer_id, order_date.isoformat(), round(computed_total, 2)))

    cursor.executemany(
        "INSERT INTO orders (id, customer_id, order_date, total_amount) VALUES (?, ?, ?, ?)",
        orders,
    )

    cursor.executemany(
        "INSERT INTO order_items (id, order_id, product_id, quantity) VALUES (?, ?, ?, ?)",
        order_items,
    )

    conn.commit()
    conn.close()
