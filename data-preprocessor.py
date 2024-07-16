import pandas as pd
import os

CHUNK_SIZE = 100  # Process 100 rows at a time
MAX_RECORDS = 10000  # Total number of extracted records

DESCRIPTION_TEXT_TO_REMOVE = "[!@@Additional Info@@!]"

# Transmission Types
transmission_types = {
    "A": "Automatic",
    "M": "Manual",
    "CVT": "Continuously Variable Transmission"
}


def extract_records(source: str, cols_list: list[str]):
    # Dataframe that stores the preprocessed results
    result = pd.DataFrame()

    for chunk in pd.read_csv(source, chunksize=CHUNK_SIZE):
        # Drop the specified cols
        chunk = chunk.drop(columns=cols_list, errors='ignore')

        # Format transmission
        if "transmission" in chunk.columns:
            chunk["transmission"] = (chunk["transmission"]
                                     .map(transmission_types)
                                     .fillna(chunk["transmission"]))

        # Format description
        if "description" in chunk.columns:
            chunk["description"] = chunk["description"].str.replace(
                DESCRIPTION_TEXT_TO_REMOVE, '', regex=False)

        result = pd.concat([result, chunk], ignore_index=True)

        if len(result) >= MAX_RECORDS:
            break

    return result


def main():
    # src - csv file to be used, dest - preprocessed csv file
    src = "data/used_cars_data.csv"
    dest = "data/used_cars_extracted.csv"

    # Columns to exclude
    excluded_cols = ["main_picture_url", "salvage", "savings_amount", "theft_title",
                     "trimId", "vehicle_damage_category", "cabin", "isCab", "is_certified",
                     "is_cpo", "is_oemcpo", "dealer_zip"]

    if os.path.exists(dest):
        os.remove(dest)

    result = extract_records(src, excluded_cols)

    # Save the dataframe as csv
    result.to_csv(dest, index=False)

    print(f"Successfully saved {len(result)} records to {dest}")


if __name__ == "__main__":
    main()
