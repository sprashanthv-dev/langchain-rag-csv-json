import argparse
import os

import openai

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

CHROMA_DIR_PATH = "./chroma"
RELEVANCE_THRESHOLD = 0.7

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


def query_rag(query: str):
    # Init vector DB
    embedding_fn = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_DIR_PATH, embedding_function=embedding_fn)

    print("Querying the vector db ....")

    # Search vector DB
    results = db.similarity_search_with_score(query, 5)

    print(f"Length of results {len(results)}")

    # if len(results) == 0 or results[0][1] < RELEVANCE_THRESHOLD:
    #     print(f"Unable to find any matching results for the query")
    #     return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)

    model = ChatOpenAI()
    response = model.invoke(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]

    formatted_response = f"Response: {response.content}\nSources: {sources}"
    print(formatted_response)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("query_text", type=str, help="Query text")
    args = parser.parse_args()

    query_text = args.query_text
    query_rag(query_text)


if __name__ == "__main__":
    main()
