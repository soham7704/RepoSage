from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

RELEVANT_EXTENSIONS = [".py", ".ipynb", ".html", ".css", ".js", ".toml",".md", ".txt", ".json", ".yml"]

def is_valid_file(filename):
    return any(filename.endswith(ext) for ext in RELEVANT_EXTENSIONS)

def load_repo_files(file_list):
    docs = []
    for file in file_list:
        if is_valid_file(file["path"]):
            doc = Document(
                page_content=file["content"],
                metadata={"source": file["path"]}
            )
            docs.append(doc)
    return docs

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)
