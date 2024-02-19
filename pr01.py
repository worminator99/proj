
#Here the necessary libraries are imported and installed
import streamlit as st
from langchain.llms import OpenAI  # this is statement that will correctly import OpenAi, had many issues importing openai, and this is the only way that it works for my computer system
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

#Defines the function to generate the response
def generate_response(txt, openai_api_key):
    try: 
    # Instantiate the LLM model
        llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
    # Split text
        text_splitter = CharacterTextSplitter()
        texts = text_splitter.split_text(txt)
    # Create multiple documents
        docs = [Document(page_content=t) for t in texts]
    # Text summarization
        chain = load_summarize_chain(llm, chain_type='map_reduce')
        response = chain.run(docs)
        return response
    except Exception as e:
        print(f"Error occurred: {e}")
        st.error(f"Error occurred: {e}")
        return f"Error: {str(e)}"

# Page title
st.set_page_config(page_title='AI Based Content Summarization')
st.title('AI Based Content Summarization App')

# Prompting the user to input the content or text they want to summarize
txt_input = st.text_area('Enter the text you want to summarize here', '', height=300)

# Summarize form that accept the user's text input for summarization
result = []
with st.form('summarize_form', clear_on_submit=True):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not txt_input)
    submitted = st.form_submit_button('Submit')

#
    if submitted:
        if not txt_input.strip():  
            st.warning("Please enter some text for summarization.")
        elif openai_api_key.startswith('sk-'):
            with st.spinner('Calculating summary...'):
                response = generate_response(txt_input, openai_api_key)
                result.append(response)
                del openai_api_key
        else:
            st.warning("Invalid OpenAI API Key. Please provide a valid key.")

#This part of the code is responsible for the output
if len(result):
    st.success('The summarization you requested completed successfully!')
    # Replace newline characters with an empty string
    cleaned_result = result[0].replace('\n', '')
    st.info(cleaned_result)



  
