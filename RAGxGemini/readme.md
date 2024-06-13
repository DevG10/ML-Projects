![alt text](https://drive.google.com/uc?export=view&id=1hgH4SnXSHDueRjOwp_G6ZlhsQwfVobTy)
#### [Link to the RAG Model](https://ragwithgemini.streamlit.app/)
## Introduction

This project aims to build a Retrieval-Augmented Generation (RAG) model using Gemini Pro with Google Generative AI. The model is designed to provide relevant answers to user queries based on a structured dataset.

## Dataset Construction
- The datset contains two columns as "Queries" which were general queries I found in the booklet and other column as "Answer" which contains the relevant answers copied from the pdf itself.
- There are total 45 entries of Query-Answer pairs.
### Source of Data

The dataset was sourced from the [link](https://assets.churchill.com/motor-docs/policy-booklet-0923.pdf) provided in the mail.

### Dataset Description
The dataset includes:

- **Queries**: Questions or queries posed by users.
- **Answers**: Corresponding answers to each query.

### Dataset Example

Example snippet from the dataset:

| Query                  | Answer                         |
|------------------------|--------------------------------|
| what is track day | Track day is when your car is driven on a racing track, on an airfield or at an off-road event. |

### Model Description

The RAG model utilizes Gemini Pro with Google Generative AI for generating responses based on retrieved documents (answers). I chose this as it wasy easy to integrate and has good answering capacities

### Evaluation Metrics
For Evaluating RAG I first used Precision, Recall and F1 score. Later I realised that they were not yeilding much information for the task. Then doing some research i found that for evaluating this kind of thing I need something as BLEU Score or ROUGE. So, I decided to go with BLEU and compared the Exact matches. 
The following evaluation metrics were chosen:

- **BLEU Score**: It is a metric for evaluating the quality of text which has been machine-translated from one language to another.
- **Exact Matches**: Comparision between the genrated answers and answers from the PDF.

### Why These Metrics?

These metrics provide a comprehensive assessment of the model's performance in retrieval and generation tasks, ensuring both relevance and completeness of answers.

## Improving Accuracy

### Steps Taken

Efforts to enhance model accuracy included:

- Experimenting with different embeddings (e.g., `all-MiniLM-L6-v2` from Sentence Transformers).
- Tuning model parameters such as temperature in generation.
- Optimizing the FAISS index configuration for efficient retrieval.

### Results and Challenges

Changes implemented led to noticeable improvements in Exact Matches. Challenges included storing and retrieving the calculated emebeddings.

## Conclusion

In conclusion, this project demonstrated the effectiveness of integrating a RAG model using Gemini Pro with Google Generative AI. Future work may focus on making the dataset much more larger for better pattern understanding by LLM, continuing to refine model performance and expand its application.

