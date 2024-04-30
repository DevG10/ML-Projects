from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder)
import streamlit as st
from streamlit_chat import message
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC
from openai import OpenAI
import os
load_dotenv()
client = OpenAI()
st.set_page_config(page_title='Ayurveda Chatbot', page_icon=':herb:')
st.header('Ayurvedic chatbot for a healthy routine :herb:')
with st.sidebar:
    st.title('Ayurveda Chatbot')
    st.markdown('''
        ## About
        Personalised Ayurveda Chatbot built using classic ayurvedic books
        ''')
    st.write('Made with ❤️ by [Team Ayurmarg](https://www.youtube.com/watch?v=6_fYrOHm7QE&t=13s)')

    st.write('Popular Books used for training:')
    st.write('''
        - [Charak Samhita](https://www.wisdomlib.org/hinduism/book/charaka-samhita-english)
        - [Sushruta Samhita](https://www.wisdomlib.org/hinduism/book/sushruta-samhita-volume-1-sutrasthana)
        - [Ashtanga Hridaya](https://www.planetayurveda.com/ayurveda-ebooks/astanga-hridaya-sutrasthan-handbook.pdf)
        
        ''')
    st.info('**Note:** As this bot is trained on Ayurveda books so it can answer questions **only related to Ayurveda**')
    st.error("This bot is **not a substitute for a doctor**. Please  don't rely solely on this bot!.")


if 'responses' not in st.session_state:
    st.session_state['responses'] = ['How can I help you?']
if 'requests' not in st.session_state:
    st.session_state['requests'] = []
    
@st.cache_resource(show_spinner=False)
def load_model():
    llm = ChatOpenAI(model_name = "gpt-3.5-turbo")
    return llm

# llm = ChatOpenAI(model_name = "gpt-3.5-turbo")

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = None
    
if st.session_state.buffer_memory is None:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)
    


    
system_msg_template = SystemMessagePromptTemplate.from_template(template = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say 'As an Ayurvedic assistant, I can only answer questions related to Ayurveda.' else if the user is askign for greeting messages you can greet him back with a greeting message.""")

human_msg_template = HumanMessagePromptTemplate.from_template(template = "{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name = "history"), human_msg_template])

if 'conversation_chain' not in st.session_state:
    st.session_state.conversation_chain = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=load_model(), verbose=False)

conversation = st.session_state.conversation_chain

# conversation = ConversationChain(memory = st.session_state.buffer_mmeory, prompt = prompt_template, llm = llm, verbose = False)

response_container = st.container()
text_container = st.container()

model = SentenceTransformer('all-MiniLM-L6-v2')

pc = PineconeGRPC(api_key=os.getenv("PINECONE_API_KEY"))

index = pc.Index('ayurveda-chatbot')

if st.session_state.buffer_memory is None:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)


@st.cache_data(show_spinner=False)
def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=20, include_metadata=True)

    if 'metadata' in result['matches'][0] and 'text' in result['matches'][0]['metadata']:
        text1 = result['matches'][0]['metadata']['text']
    else:
        text1 = "No text in metadata for match 1"

    if 'metadata' in result['matches'][1] and 'text' in result['matches'][1]['metadata']:
        text2 = result['matches'][1]['metadata']['text']
    else:
        text2 = "No text in metadata for match 2"

    return text1 + "\n" + text2


@st.cache_data(show_spinner=False)
def query_refiner(conversation, query):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a highly helpful assistant providing information only on ayurveda related topics and you can answer the in general questions like hi, hello and other type of greetings."},
        {"role": "user", "content": f"Context: {conversation}"},
        {"role": "assistant", "content": f"Query: {query}"}],

        temperature=0.5,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    st.spinner(text="")
    return response.choices[0].message.content

def get_conversation_string():
    conversation_string = []
    for i in range(len(st.session_state['responses']) - 1):        
        conversation_string.append("Human: "+st.session_state['requests'][i])
        conversation_string.append("Bot: "+ st.session_state['responses'][i+1])
    return '\n'.join(conversation_string)

if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

with text_container:
    query = st.text_input("How may I help you? ", key="input")
    if query:
        with st.spinner("Generating personalized response..."):
            conversation_string = get_conversation_string()
            # st.code(conversation_string)
            refined_query = query_refiner(conversation_string, query)
            context = find_match(refined_query)
            response = conversation.predict(input=f"Context: \n {context} \n\n Query: \n {query}")
            st.session_state.requests.append(query)
            st.session_state.responses.append(response)

with response_container:
    if st.session_state['responses']:

        
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key = str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state['requests'][i], is_user = True, key = str(i) + '_user')
                message(st.session_state['requests'][i], is_user = True, key = str(i) + '_user')
