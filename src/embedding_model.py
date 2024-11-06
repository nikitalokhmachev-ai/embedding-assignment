from typing import List, Dict, Any
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import re

from src.utils.datetime_parser import parse_datetime  
from src.utils.action_inference import find_action  

class EmbeddingModel:
    def __init__(self, dataset: pd.DataFrame):
        # Initialize tokenizer and model for embeddings
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
        # Define dataset and infer column types
        self.dataset = dataset
        self.columns = self.infer_column_types(dataset)

        # Pre-compute embeddings for column names
        self.column_embeddings = self.compute_embeddings(list(self.columns.keys()))
        
        # Pre-compute embeddings for categorical values
        self.value_embeddings = {}
        for column, col_type in self.columns.items():
            if col_type == "category":
                unique_values = dataset[column].dropna().unique().tolist()
                self.value_embeddings[column] = self.compute_embeddings(unique_values)
        
        # Define actions and pre-compute embeddings for them
        self.actions = {
            "date": ["before date", "after date", "on date", "between dates"],
            "numeric": ["greater than", "less than", "equals to", "equal to", "between", "maximum", "minimum", "most frequent"],
        }
        self.action_embeddings = {
            action_type: self.compute_embeddings(actions)
            for action_type, actions in self.actions.items()
        }

    def infer_column_types(self, dataset: pd.DataFrame) -> Dict[str, str]:
        """Infers column types as date, numeric, or category."""
        column_types = {}
        for column in dataset.columns:
            dtype = dataset[column].dtype
            if pd.api.types.is_datetime64_any_dtype(dtype):
                column_types[column] = "date"
            elif pd.api.types.is_numeric_dtype(dtype):
                column_types[column] = "numeric"
            elif pd.api.types.is_object_dtype(dtype):
                column_types[column] = "category"
            else:
                column_types[column] = "category"
        return column_types

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encodes a list of text inputs into embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        encoded_input = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            model_output = self.model(**encoded_input).last_hidden_state
        embeddings = model_output.mean(dim=1)  # Mean pooling
        return embeddings

    def compute_embeddings(self, items: List[str]) -> Dict[str, torch.Tensor]:
        """Computes embeddings for a list of items in a single batch."""
        embeddings = self.encode_texts(items)
        return {item: embeddings[i] for i, item in enumerate(items)}
    
    def infer_column(self, query: str) -> str:
        """Infers the column name based on closest match."""
        column_names = list(self.column_embeddings.keys())
        column_embeddings_matrix = torch.stack(list(self.column_embeddings.values()))
        query_embedding = self.encode_texts(query).expand(column_embeddings_matrix.size(0), -1)
        similarities = torch.nn.functional.cosine_similarity(column_embeddings_matrix, query_embedding, dim=1)
        best_match_index = similarities.argmax().item()
        return column_names[best_match_index]

    def infer_action(self, query: str, column_type: str) -> str:
        """Infers the action based on closest matching action embedding."""
        if column_type == "category":
            return None
        inferred_action = find_action(self.actions[column_type], query)
        if inferred_action:
            return inferred_action
        query_embedding = self.encode_texts(query)
        action_embeddings = self.action_embeddings[column_type]
        action_names = list(action_embeddings.keys())
        action_embeddings_matrix = torch.stack(list(action_embeddings.values()))
        similarities = torch.nn.functional.cosine_similarity(action_embeddings_matrix, query_embedding.expand(action_embeddings_matrix.size(0), -1), dim=1)
        best_action_index = similarities.argmax().item()
        return action_names[best_action_index]

    def infer_date_value(self, query: str) -> Any:
        """Infers date values or ranges from the query without returning action."""
        parsed_dates = parse_datetime(query)
        if len(parsed_dates) == 2:
            return parsed_dates[0].date(), parsed_dates[1].date()
        elif parsed_dates:
            return parsed_dates[0].date()
        return None

    def infer_numeric_value(self, query: str, column: str) -> Any:
        """Infers numeric values or ranges from the query without returning action."""
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", query)
        if len(numbers) == 2:
            return float(numbers[0]), float(numbers[1])
        elif numbers:
            return float(numbers[0])
        return None

    def infer_category_value(self, query: str, column: str) -> Any:
        """Infers the closest matching category value using embeddings without returning action."""
        query_embedding = self.encode_texts(query)
        value_embeddings = self.value_embeddings[column]
        value_names = list(value_embeddings.keys())
        value_embeddings_matrix = torch.stack(list(value_embeddings.values()))
        similarities = torch.nn.functional.cosine_similarity(value_embeddings_matrix, query_embedding.expand(value_embeddings_matrix.size(0), -1), dim=1)
        best_value_index = similarities.argmax().item()
        return value_names[best_value_index]

    def infer_value(self, query: str, column: str) -> Any:
        """Directs to specific value inference methods based on column type, returning only the value."""
        col_type = self.columns[column]
        if col_type == "date":
            return self.infer_date_value(query)
        elif col_type == "numeric":
            return self.infer_numeric_value(query, column)
        elif col_type == "category":
            return self.infer_category_value(query, column)
        return None

    def filter_column(self, column: str, value: Any, action: str) -> List[int]:
        """Filters the dataset based on the column type, inferred value, and the query for actions."""
        col_type = self.columns[column]
        if col_type == "date":
            if action == "before date":
                return self.dataset[self.dataset[column].dt.date < value].index.tolist()
            elif action == "after date":
                return self.dataset[self.dataset[column].dt.date > value].index.tolist()
            elif action == "on date":
                return self.dataset[self.dataset[column].dt.date == value].index.tolist()
            elif action == "between dates" and isinstance(value, tuple):
                start_date, end_date = value
                return self.dataset[(self.dataset[column].dt.date >= start_date) & (self.dataset[column].dt.date <= end_date)].index.tolist()
        elif col_type == "numeric":
            if action == "greater than":
                return self.dataset[self.dataset[column] > value].index.tolist()
            elif action == "less than":
                return self.dataset[self.dataset[column] < value].index.tolist()
            elif action == "equal to" or action == "is" or action == "equals to":
                return self.dataset[self.dataset[column] == value].index.tolist()
            elif action == "between" and isinstance(value, tuple):
                min_val, max_val = value
                return self.dataset[(self.dataset[column] >= min_val) & (self.dataset[column] <= max_val)].index.tolist()
            elif action == "most frequent":
                most_frequent_value = self.dataset[column].mode()[0]
                return self.dataset[self.dataset[column] == most_frequent_value].index.tolist()
            elif action == "maximum":
                max_value = self.dataset[column].max()
                return self.dataset[self.dataset[column] == max_value].index.tolist()
            elif action == "minimum":
                min_value = self.dataset[column].min()
                return self.dataset[self.dataset[column] == min_value].index.tolist()
        elif col_type == "category":
            return self.dataset[self.dataset[column] == value].index.tolist()
        return []

    def search(self, query: str) -> Dict:
        """Main search function to interpret query and filter results."""
        inferred_column = self.infer_column(query)
        inferred_value = self.infer_value(query, inferred_column)
        action = self.infer_action(query, self.columns[inferred_column])
        row_ids = self.filter_column(inferred_column, inferred_value, action)
        return {
            "column_name": inferred_column,
            "value": inferred_value,
            "row_ids": row_ids,
            "action": action
        }