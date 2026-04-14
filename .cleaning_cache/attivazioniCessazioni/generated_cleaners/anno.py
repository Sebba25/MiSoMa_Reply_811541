import re

def clean_anno(value):
    if value is None:
        return None
    value_str = str(value).strip()
    cleaned_value = re.sub(r'\D', '', value_str)[:4]
    if len(cleaned_value) == 4:
        return cleaned_value
    return value_str
