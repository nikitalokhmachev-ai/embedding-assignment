import re
import dateparser

def parse_datetime(text):
    """
    Extracts and parses dates from a sentence-level string.
    
    Args:
        text (str): The input text containing date phrases.

    Returns:
        list: A list of datetime objects for all detected dates in the text.
    """
    # Define regex patterns to identify date-like phrases within sentences
    date_patterns = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',          # 12-31-2022, 12/31/22
        r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',            # 2022-12-31, 2022/12/31
        r'\b\d{1,2}[.]\d{1,2}[.]\d{2,4}\b',            # 31.12.2022
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}[,]?\s+\d{2,4}\b',  # Dec 31, 2022
        r'\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b',      # 31 Dec 2022
        r'\b(?:today|yesterday|tomorrow|last\s+week|next\s+week)\b',  # Natural language dates
    ]
    
    # Compile regex pattern for sentence-level date-like phrases
    date_regex = re.compile('|'.join(date_patterns), re.IGNORECASE)
    
    # Find all date-like phrases in the text
    potential_date_phrases = date_regex.findall(text)
    
    # Parse each detected phrase with dateparser
    parsed_dates = []
    for phrase in potential_date_phrases:
        parsed_date = dateparser.parse(phrase)
        if parsed_date:
            parsed_dates.append(parsed_date)
    
    # Remove duplicates and return sorted dates
    unique_dates = sorted(set(parsed_dates))
    return unique_dates