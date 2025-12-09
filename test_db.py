from utils.database import run_query

print("Tables:", run_query("SELECT name FROM sqlite_master WHERE type='table'"))

rows = run_query("SELECT * FROM service_plans LIMIT 3")
print("\nRows:")
for r in rows:
    print(r)
