from langchain_community.embeddings.ollama import OllamaEmbeddings

EMBEDDING_MODEL = "nomic-embed-text"


def get_embedding_fn():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return embeddings
