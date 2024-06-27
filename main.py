import os
import streamlit as st
import pickle
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()


# Load the Gemini API key
genai.configure(api_key=os.environ['GEMINI_API_KEY'])
model = genai.GenerativeModel('gemini-1.5-flash')

# Streamlit UI
st.title("Bull's Eye: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_gemini.pkl"  # Changed filename to avoid conflict
main_placeholder = st.empty()

# Define LLM function
def generate_response(prompt):
    try:
        response = model.generate_text(
            prompt=prompt,
            temperature=0.9,
            max_output_tokens=500
        )
        return response.result
    except Exception as e:
        st.error(f"Error generating response from Gemini: {e}")
        return "Error occurred while processing your request."  # Handle errors gracefully

# Main processing logic
if process_url_clicked:
    try:
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("Data Loading...Started...✅✅✅")
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("Text Splitter...Started...✅✅✅")
        docs = text_splitter.split_documents(data)

        # No need for embeddings with Gemini, as it doesn't use them
        vectorstore_gemini = FAISS.from_documents(docs)
        main_placeholder.text("Vector Store Built...✅✅✅")
        time.sleep(2)

        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_gemini, f)

    except Exception as e:
        st.error(f"Error processing URLs: {e}")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)

            chain = RetrievalQAWithSourcesChain.from_chain_type(llm=generate_response, chain_type="stuff", retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)

            st.header("Answer")
            st.write(result["answer"])

            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)
        except Exception as e:
            st.error(f"Error retrieving answer: {e}")
