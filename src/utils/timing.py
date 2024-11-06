import time
def calculate_token_processing_speed(queries, embedding_model):
    """
    Calculates and returns the token processing speed (tokens per second) for each query in `queries`.
    """
    start_time = time.time()
    for query in queries:
        # Let's assume here that word count is token count.
        embedding_model.search(query)
    end_time = time.time()

    total_tokens = 0
    for query in queries:
        # Counting tokens
        num_tokens = len(query.split())
        total_tokens += num_tokens

    
    overall_processing_speed = total_tokens / (end_time - start_time)
    
    return overall_processing_speed