from datetime import datetime

def aggregation_time(value):
    if value is None:
        return None
    
    # Attempt to parse the input value into the expected format
    formats = [
        "%Y-%m-%dT%H:%M:%S.%f",  # ISO format with microseconds
        "%Y-%m-%dT%H:%M:%S",      # ISO format without microseconds
        "%d/%m/%Y",                # European format
        "%d.%m.%Y",                # European format with dots
        "%d %b %Y",                # Month abbreviation
        "%Y/%m/%d",                # Year/Month/Day format
        "%d-%m-%y"                 # Short year format
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(value, fmt)
            return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
        except ValueError:
            continue
    
    # If no format matched, return the original value
    return value
