import pandas as pd
import os

from col_helpers import get_excluded_cols, construct_sentence
from pdf_helper import PDFHelper

CHUNK_SIZE = 10  # Process 5 rows at a time
MAX_RECORDS = 100  # Total number of extracted records

# src - csv file to be used, dest - preprocessed csv file
src_csv = "data/used_cars_data.csv"
dest_csv = "data/extracted/used_cars_extracted.csv"
dest_pdf = "data/extracted/used_cars_extracted.pdf"

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

    text = text + "\n\n".join(sentences)

    return text


def main():
    # Columns to exclude
    excluded_cols = get_excluded_cols()

    if os.path.exists(dest_csv):
        os.remove(dest_csv)

    if os.path.exists(dest_pdf):
        os.remove(dest_pdf)

    result = extract_records(src_csv, excluded_cols)

    # Save the dataframe as csv
    result.to_csv(dest_csv, index=False)

    print(f"Successfully saved {len(result)} records to {dest_csv}")

    text = construct_sentences(result)

    # Save the constructed sentences as a PDF document
    pdf = PDFHelper()
    pdf.add_chapter(text)

    pdf.output(dest_pdf)
    print(f"Successfully saved {len(result)} records to {dest_pdf}")


if __name__ == "__main__":
    main()
