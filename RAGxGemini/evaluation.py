import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from langchain_community.vectorstores import FAISS
from backend import get_conversation_chain
import time
from langchain_community.embeddings import HuggingFaceEmbeddings
import warnings
warnings.filterwarnings('ignore')

# Load evaluation dataset
eval_data = pd.read_excel('dataset_rag.xlsx')
queries = eval_data['Queries'].tolist()
ground_truth_answers = eval_data['Answer'].tolist()

# Initialize embeddings and FAISS index
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
vector_data = FAISS.load_local('faiss_indexes', embeddings, allow_dangerous_deserialization = True)



def evaluate_model(queries, ground_truth_answers):
    chain = get_conversation_chain()
    bleu_scores = []
    exact_matches = []

    for query, true_answer in zip(queries, ground_truth_answers):
        query_embedding = embeddings.embed_query(query)
        docs = vector_data.similarity_search_by_vector(query_embedding)
        response = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
        predicted_answer = response['output_text']

        # BLEU Score
        bleu = sentence_bleu([true_answer.split()], predicted_answer.split())
        bleu_scores.append(bleu)

        # Exact Match
        exact_matches.append(int(true_answer.strip().lower() == predicted_answer.strip().lower()))
        time.sleep(2)
    results = {
        'BLEU Score': sum(bleu_scores) / len(bleu_scores),
        'Exact Match': sum(exact_matches) / len(exact_matches)
    }

    return results


# Run evaluation
results = evaluate_model(queries, ground_truth_answers)
print(results)
