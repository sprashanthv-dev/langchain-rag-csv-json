import argparse

from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

from embeddings import get_embedding_fn

CHROMA_DIR_PATH = "./chroma"
LLM_MODEL = "mistral"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def query_rag(query: str):
    # Init vector DB
    embedding_fn = get_embedding_fn()
    db = Chroma(persist_directory=CHROMA_DIR_PATH, embedding_function=embedding_fn)

    # Search vector DB
    results = db.similarity_search_with_score(query, 5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)

    model = Ollama(model=LLM_MODEL)
    response = model.invoke(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]

    formatted_response = f"Response: {response}\nSources: {sources}"
    print(formatted_response)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("query_text", type=str, help="Query text")
    args = parser.parse_args()

    query_text = args.query_text
    query_rag(query_text)


if __name__ == "__main__":
    main()
