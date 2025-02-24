from flask import Flask,jsonify,render_template,request
from src.helper import dowmload_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app=Flask(__name__)
load_dotenv()
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
os.environ['PINECONE_API_KEY']=PINECONE_API_KEY
embeddings=dowmload_embeddings()
index_name='medicalbot'
docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
    )
retreiever=docsearch.as_retriever(search_type='similarity',search_kwargs={'k':3})
llm=OllamaLLM(model='llama3',temperature=0,max_tokens=500)
prompt=ChatPromptTemplate.from_messages([
    ("system",system_prompt),
    ("human","{input}")
])
question_answer_chain=create_stuff_documents_chain(llm=llm,prompt=prompt)
rag_chain=create_retrieval_chain(retreiever,question_answer_chain)
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/get',methods=["GET","POST"])
def chat():
    msg=request.form.get("msg")
    input=msg
    print(input)
    response=rag_chain.invoke({'input':msg})
    print("Response",response['answer'])
    return str(response['answer'])
if __name__ =="__main__":
    app.run(host='0.0.0.0',port=8080,debug=True)