def clean_mese(value):
    if value is None:
        return None
    
    # Define month mappings
    month_map = {
        "1": "January", "2": "February", "3": "March", "4": "April",
        "5": "May", "6": "June", "7": "July", "8": "August",
        "9": "September", "10": "October", "11": "November", "12": "December",
        "JAN": "January", "FEB": "February", "MAR": "March", "APR": "April",
        "MAY": "May", "JUN": "June", "JUL": "July", "AUG": "August",
        "SEP": "September", "OCT": "October", "NOV": "November", "DEC": "December",
        "Marzo": "March", "Maggio": "May", "Dicembre": "December",
        "Ottobre": "October", "Settembre": "September"
    }
    
    # Normalize the input
    value_str = str(value).strip()
    
    # Check if the value is a valid month number
    if value_str.isdigit() and 1 <= int(value_str) <= 12:
        return value_str  # Return as is, it's already a valid month number
    
    # Check if the value is a valid month name
    if value_str in month_map:
        return month_map[value_str]  # Return the full month name
    
    # If the value is not valid, return it unchanged
    return value_str
