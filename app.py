import time
import os
import joblib
import json
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from google.cloud import speech
from google.oauth2 import service_account
import base64

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
SERVICE_ACCOUNT_JSON = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')

# Configure generative AI
genai.configure(api_key=GOOGLE_API_KEY)

# Create a data/ folder if it doesn't already exist
try:
    os.mkdir('data/')
except FileExistsError:
    pass

# Load past chats (if available)
try:
    past_chats = joblib.load('data/past_chats_list')
except FileNotFoundError:
    past_chats = {}

# Sidebar for past chats
with st.sidebar:
    st.image("assets/duckGPT.png")
    st.write('# Past Chats')
    chat_id = st.selectbox(
        label='Pick a past chat',
        options=['New Chat'] + list(past_chats.keys()),
        index=0,
        format_func=lambda x: 'New Chat' if x == 'New Chat' else past_chats.get(x, 'New Chat'),
    )
    if chat_id == 'New Chat':
        chat_id = f'{time.time()}'
    st.session_state['chat_id'] = chat_id
    st.session_state['chat_title'] = past_chats.get(chat_id, f'ChatSession-{chat_id}')

st.write('# Chat with DuckGPT')

# Chat history (allows asking multiple questions)
try:
    st.session_state['messages'] = joblib.load(f'data/{chat_id}-st_messages')
    st.session_state['gemini_history'] = joblib.load(f'data/{chat_id}-gemini_messages')
except FileNotFoundError:
    st.session_state['messages'] = []
    st.session_state['gemini_history'] = []

# Configure Google Cloud Speech client
print("decoding")
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_JSON)
speech_client = speech.SpeechClient(credentials=credentials)

# Function to transcribe audio
def transcribe_audio(audio_data):
    audio = speech.RecognitionAudio(content=audio_data)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )
    response = speech_client.recognize(config=config, audio=audio)
    return response.results[0].alternatives[0].transcript if response.results else ""

# Function to handle user input
def handle_user_input(prompt):
    # Save new chats after a message has been sent to AI
    if chat_id not in past_chats.keys():
        past_chats[chat_id] = st.session_state['chat_title']
        joblib.dump(past_chats, 'data/past_chats_list')
    
    # Display user message in chat message container
    with st.chat_message('user'):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state['messages'].append({'role': 'user', 'content': prompt})
    
    # Send message to AI
    response = st.session_state['chat'].send_message(prompt, stream=True)
    
    # Display assistant response in chat message container
    with st.chat_message('duck', avatar='🦆'):
        message_placeholder = st.empty()
        full_response = ''
        for chunk in response:
            for ch in chunk.text.split(' '):
                full_response += ch + ' '
                time.sleep(0.05)
                message_placeholder.write(full_response + '▌')
        message_placeholder.write(full_response)
    
    # Add assistant response to chat history
    st.session_state['messages'].append({'role': 'duck', 'content': full_response, 'avatar': '🦆'})
    st.session_state['gemini_history'] = st.session_state['chat'].history
    
    # Save to file
    joblib.dump(st.session_state['messages'], f'data/{chat_id}-st_messages')
    joblib.dump(st.session_state['gemini_history'], f'data/{chat_id}-gemini_messages')

# Audio input
audio_input = st.audio_input('Record your message')
if audio_input is not None:
    audio_bytes = audio_input.read()
    transcript = transcribe_audio(audio_bytes)
    if transcript:
        st.write("You said: ", transcript)
        handle_user_input(transcript)
