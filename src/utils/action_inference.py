def find_action(strings, target):
    """
    Finds the value of the first substring in the given list of strings that appears in the target string.
    The search is case-insensitive.
    """
    target_lower = target.lower()
    for s in strings:
        if s.lower() in target_lower:
            return s
