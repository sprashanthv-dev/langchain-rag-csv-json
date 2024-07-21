import os
import openai
import shutil

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma

from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv


CSV_FILE_PATH = "./data/extracted/used_cars_extracted.csv"
PDF_DIR_PATH = "./data/pdf"
CHROMA_DB_PATH = "./chroma"

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']


def load_docs(is_pdf: bool):

    if is_pdf:
        pdf_loader = PyPDFDirectoryLoader(PDF_DIR_PATH)
        return pdf_loader.load()

    csv_loader = CSVLoader(file_path=CSV_FILE_PATH)
    return csv_loader.load()


def split_docs(docs: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False
    )

    chunks = splitter.split_documents(docs)
    print(f"Split {len(docs)} documents into {len(chunks)} chunks")

    return chunks


def calculate_chunk_ids(chunks: list[Document]):
    last_page_id = None
    current_chunk_idx = 0

    for chunk in chunks:
        src = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        page_id = f"{src}:{page}"

        # Multiple chunks share the same page
        if page_id == last_page_id:
            current_chunk_idx += 1
        # We are on a new page (i.e) all chunks within a page have been processed
        else:
            current_chunk_idx = 0

        chunk_id = f"{page_id}:{current_chunk_idx}"
        last_page_id = page_id

        chunk.metadata["id"] = chunk_id

    return chunks


def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)

    db = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(),
        persist_directory=CHROMA_DB_PATH
    )

    db.persist()
    print(f"Saved {len(chunks)} documents to {CHROMA_DB_PATH}")


def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=OpenAIEmbeddings())

    updated_chunks = calculate_chunk_ids(chunks)

    # Query db for all items (ids are included by default)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])

    print(f"Number of existing documents in the db: {len(existing_ids)}")

    new_chunks = []

    for chunk in updated_chunks:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks) > 0:
        print(f"Adding {len(new_chunks)} documents to the db")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")


def main():
    # True - If docs ingested are of pdf type, false otherwise
    is_pdf = True

    docs = load_docs(is_pdf)

    # If pdf docs are ingested, we need to chunk it for better performance
    # before saving it to our vector database
    if is_pdf:
        chunks = split_docs(docs)
        add_to_chroma(chunks)
    else:
        save_to_chroma(docs)


if __name__ == "__main__":
    main()
