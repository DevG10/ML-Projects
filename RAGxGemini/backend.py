import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
import warnings
warnings.filterwarnings('ignore')

# Initializing the Gemini Model
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

#The below function work was to calculate the embeddings and save it locally. THis was for one time use that is why I have commented it out. Uncomment if you are running code for the first time.

# Reading the created dataset
# data = pd.read_excel('dataset_rag.xlsx')
# answers = data['Answer'].tolist()
#
#
# def get_vector_store(text):
#     embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
#     vector_store = FAISS.from_texts(text, embedding=embeddings)
#     vector_store.save_local("faiss_indexes")
#
#
# get_vector_store([words for words in answers])

def get_conversation_chain():
    prompt_template = """
    Answer the question given from the provided context and make sure to provide the answers which are in the context don't try to make your own answers. If the answer is not in the context then simply say I don't know about that as it is not in the given context.\n
     Context: \n{context}\n
     Question: \n{question}\n
     """
    model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.4)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(question):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_data = FAISS.load_local('RAGxGemini/faiss_indexes', embeddings, allow_dangerous_deserialization=True)
    query_embedding = embeddings.embed_query(question)
    docs = vector_data.similarity_search_by_vector(query_embedding)
    chain = get_conversation_chain()
    response = chain(
        {"input_documents": docs, "question": question},
        return_only_outputs=True
    )
    print(response)


if __name__ == "__main__":
    question = "I want to change my policy"
    user_input(question)
