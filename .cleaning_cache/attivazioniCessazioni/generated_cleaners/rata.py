def clean_rata(value):
    if value is None:
        return None
    if isinstance(value, str):
        # Remove any non-numeric characters
        cleaned_value = ''.join(filter(str.isdigit, value))
        # Check if the cleaned value matches the expected pattern
        if len(cleaned_value) == 6 and cleaned_value.isdigit():
            return cleaned_value
    return value  # Preserve the original value if it cannot be normalized
