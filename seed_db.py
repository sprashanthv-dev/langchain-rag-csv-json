from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma

from langchain.schema import Document

import os
import shutil

from embeddings import get_embedding_fn

CSV_FILE_PATH = "./data/extracted/used_cars_extracted.csv"
CHROMA_DB_PATH = "./chroma"


def load_docs():
    csv_loader = CSVLoader(file_path=CSV_FILE_PATH)
    return csv_loader.load()


def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)

    db = Chroma.from_documents(
        chunks,
        get_embedding_fn(),
        persist_directory=CHROMA_DB_PATH
    )

    db.persist()

    print(f"Saved {len(chunks)} documents to {CHROMA_DB_PATH}")


def main():
    docs = load_docs()
    save_to_chroma(docs)


if __name__ == "__main__":
    main()
