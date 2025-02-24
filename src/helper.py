from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import  RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

#extract data from pdf file
def load_data(data):
    loader=DirectoryLoader(data,glob="*.pdf",loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents
##split the data into chunks
def text_Split(extracted_data):
    text_Splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
    text=text_Splitter.split_documents(extracted_data)
    return text
def dowmload_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

