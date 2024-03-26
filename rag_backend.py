#1. Import OS, Document Loader, Text Splitter, Bedrock Embeddings, Vector DB, VectorStoreIndex, Bedrock-LLM
import os
import boto3
from langchain.document_loaders import S3FileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms.bedrock import Bedrock

#5c. Wrap within a function
def create_index():
    #2. Define the data source and load data with PDFLoader
    s3_client = boto3.client('s3')
    
    data_load=S3FileLoader(bucket='insurance-assistance-sam', key='source/Questions_and_Answers_About_Health_Insurance.pdf')
    #data_test=data_load.load_and_split()
    #print(data_test[2])
    print('Data load completed')
    #3. Split the Text based on Character, Tokens etc. - Recursively split by character - ["\n\n", "\n", " ", ""]

    data_split=RecursiveCharacterTextSplitter(separators=["\n\n","\n", " ",""],chunk_size=100,chunk_overlap=10)
    #sample_data='You need to run the cmd prompt from the Scripts directory of Anaconda where ever you have the Anaconda parent folder installed. I happen to have in the root directory of the C drive on my Windows machine. If you are not familiar there are two ways to do that'
    #data_split_test=data_split.split_text(sample_data)
    #print(data_split_test)
    
    #4. Create Embeddings -- Client connection
    data_embeddings=BedrockEmbeddings(
        credentials_profile_name='default',
        model_id='amazon.titan-embed-text-v1')
    print('Data embedding client connection completed')
    #5à Create Vector DB, Store Embeddings and Index for Search - VectorstoreIndexCreator
    data_index=VectorstoreIndexCreator(
        text_splitter=data_split,
        embedding=data_embeddings,
        vectorstore_cls=FAISS)
    print('Vector DB created')
    #5b Create index for HR Report
    db_index=data_index.from_loaders([data_load])

    print('Indexing completed')
    print(db_index)
    return db_index

#6a. Write a function to connect to Bedrock Foundation Model
def get_llm():
    llm=Bedrock(
        credentials_profile_name='default',
        model_id='anthropic.claude-v2',
        model_kwargs={
            "max_tokens_to_sample":3000,
            "temperature":0.1,
            "top_p":0.9
        })
    return llm

#6b. Write a function which searches the user prompt, searches the best match from Vector DB and sends both to LLM.
def get_rag_response(index,question):
    rag_llm=get_llm()
    print('LLM invoked')
    hr_rag_query=index.query(question=question,llm=rag_llm)
    return hr_rag_query
# Index creation --> https://api.python.langchain.com/en/latest/indexes/langchain.indexes.vectorstore.VectorstoreIndexCreator.html