import pandas as pd
import numpy as np
from src.embedding_model import EmbeddingModel
import json 
import time

queries = [
    # Numeric Queries (for `Delivery_distance`)
    "maximum delivery distance",
    "minimum delivery distance",
    "most frequent delivery distance",
    "distances greater than 200",
    "distances less than 150",
    "distance between 100 and 300",

    # Category Queries (for `Product_Category` or `Priority`)
    "apparel product",
    "electronics items",
    "highest priority",
    "low priority shipments",
    "clothing product",
    "all medium priority orders",

    # Date Queries (for `Date`)
    "on date 2023-07-01",
    "before date 2023-08-01",
    "after date 2023-07-01",
    "between dates 2023-07-01 and 2023-07-08",
]

data = pd.read_csv('data.csv', sep=';')
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
columns = ["Date", "Priority", "Product_Category", "Delivery_distance"]
data = data[columns]
embedding_model = EmbeddingModel(data)

start = time.time()
search_results = {}
for query in queries:
    search_result = embedding_model.search(query)
    # Store the result in the dictionary with query as the key
    search_results[query] = {
        "query": query,
        "column_name": search_result["column_name"],
        "value": search_result["value"],
        "action": search_result["action"],
        "results": search_result["row_ids"]
    }

end = time.time()

# Save the search results dictionary to a JSON file
with open("search_results.json", "w") as f:
    json.dump(search_results, f, indent=4, default=str)

print("Search results have been saved to 'search_results.json'")
print(f"Time taken: {end - start} seconds")
print("Average token processing speed:", len(" ".join(queries).split(" "))/(end - start))