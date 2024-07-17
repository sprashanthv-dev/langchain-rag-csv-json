import pandas as pd
import os

from col_helpers import get_excluded_cols, construct_sentence

CHUNK_SIZE = 5  # Process 5 rows at a time
MAX_RECORDS = 25  # Total number of extracted records

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


def construct_sentences(df: pd.DataFrame):

    sentences = []
    text = ""

    for index, row in df.iterrows():
        sentence = construct_sentence(index, row)
        sentences.append(sentence)

    text = text + "\n".join(sentences)

    return text


def main():
    # src - csv file to be used, dest - preprocessed csv file
    src = "data/used_cars_data.csv"
    dest = "data/extracted/used_cars_extracted.csv"

    # Columns to exclude
    excluded_cols = get_excluded_cols()

    if os.path.exists(dest):
        os.remove(dest)

    result = extract_records(src, excluded_cols)

    # Save the dataframe as csv
    result.to_csv(dest, index=False)

    print(f"Successfully saved {len(result)} records to {dest}")

    sentences = construct_sentences(result)


if __name__ == "__main__":
    main()
