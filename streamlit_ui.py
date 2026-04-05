import streamlit as st
import requests

# --- Configuration ---
# Match this to the port your FastAPI server is running on
API_URL = "http://localhost:8000/query"

# Page setup
st.set_page_config(
    page_title="RAG Knowledge Base",
    page_icon="🤖",
    layout="centered"
)

# --- UI Header ---
st.title("🤖 Enterprise RAG Search")
st.markdown("Ask questions about the company knowledge base. The AI will answer based *only* on the ingested documents.")
st.divider()

# --- Search Interface ---
# We use a form so the user can hit 'Enter' to submit
with st.form(key="search_form"):
    question = st.text_input("What would you like to know?", placeholder="e.g., Who is the CEO of QuantumForge?")
    submit_button = st.form_submit_button(label="Search")

# --- Process Request ---
if submit_button:
    if not question.strip():
        st.warning("Please enter a question to search.")
    else:
        with st.spinner("Searching the knowledge base..."):
            try:
                # 1. Call the FastAPI backend
                response = requests.post(
                    API_URL, 
                    json={"question": question},
                    timeout=30 # Add a timeout so the UI doesn't hang forever
                )
                
                # 2. Handle successful response
                if response.status_code == 200:
                    data = response.json()
                    
                    # Display the LLM's Answer
                    st.subheader("Answer")
                    st.success(data.get("answer", "No answer provided."))
                    
                    # Display the Retrieved Sources in a collapsible expander
                    sources = data.get("sources", [])
                    if sources:
                        with st.expander(f"View Sources ({len(sources)} chunks retrieved)"):
                            for i, source in enumerate(sources):
                                st.markdown(f"**📄 {source.get('source', 'Unknown')} (Page {source.get('page', '?')})**")
                                st.caption(f"*Relevance Score: {source['score']:.4f}*")
                                st.write(source['text'])
                                if i < len(sources) - 1:
                                    st.divider()
                    else:
                        st.info("No relevant sources found in the database.")
                        
                # 3. Handle API errors
                else:
                    st.error(f"Backend Error (Status Code: {response.status_code})")
                    st.code(response.text)
                    
            except requests.exceptions.ConnectionError:
                st.error("🚨 Failed to connect to the backend API. Please ensure your FastAPI server is running on `http://localhost:8000`.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")