def clean_attivazioni(value):
    if value is None:
        return None
    match = re.match(r'(\d+)', str(value))
    if match:
        return match.group(1)
    return value
