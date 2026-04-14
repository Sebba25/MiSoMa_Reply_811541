def clean_cessazioni(value):
    if value is None:
        return None
    if isinstance(value, str):
        # Remove any non-numeric characters and whitespace
        cleaned_value = ''.join(filter(str.isdigit, value))
        return cleaned_value if cleaned_value else value
    return str(value) if isinstance(value, (int, float)) else value
