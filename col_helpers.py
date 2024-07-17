import pandas as pd

from cols_config import excluded_cols, col_mappings


def get_excluded_cols():
    return excluded_cols


def construct_sentence(index, row: pd.Series):
    # Stores different parts of the sentence
    parts = []

    # Stores various attributes of the car
    attributes = []

    intro_text = f"The {row['make_name']} {row['model_name']} has a vehicle identification number of {row['vin']}. "
    parts.append(intro_text)

    cols_list = list(col_mappings.keys())

    for col in cols_list:
        if pd.notna(row[col]):
            feature = f"{col_mappings[col]}{row[col]}. "
            attributes.append(feature)

    if attributes:
        parts.append("".join(attributes))

    sentence = "".join(parts) + ""

    return sentence
