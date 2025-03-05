import streamlit as st
import os
import tempfile
from database.vector_store import VectorDatabase
from chat.openai_client import OpenAIClient
from prompts.prompts import get_system_prompt, format_user_prompt, format_retrieved_context
from chat.conversation_handler import ConversationHandler

# Set page configuration
st.set_page_config(
    page_title="PDF Q&A with Vector Search",
    page_icon="📚",
    layout="wide"
)

# Add CSS for styling the collapsible sections
st.markdown("""
<style>
    .sources-expander {
        margin-top: 10px;
        margin-bottom: 15px;
    }
    .source-content {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_role" not in st.session_state:
    st.session_state.selected_role = "standard"
if "conversation_handler" not in st.session_state:
    st.session_state.conversation_handler = ConversationHandler()

# Function to initialize vector database
def initialize_vector_db():
    vector_db = VectorDatabase(
        collection_name="streamlit_pdf_db",
        persist_directory="./streamlit_chroma_db"
    )
    return vector_db

# Function to process uploaded PDF
def process_uploaded_pdf(uploaded_file):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        # Initialize vector database if not already done
        if st.session_state.vector_db is None:
            st.session_state.vector_db = initialize_vector_db()
        
        # Process the PDF
        with st.spinner("Processing PDF... This may take a minute."):
            chunk_ids, pdf_info = st.session_state.vector_db.process_pdf(tmp_file_path)
            st.session_state.pdf_processed = True
            st.session_state.pdf_name = uploaded_file.name
            return True, f"Successfully processed {uploaded_file.name} into {len(chunk_ids)} chunks"
    except Exception as e:
        return False, f"Error processing PDF: {str(e)}"
    finally:
        # Remove temp file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

# Function to search the vector database
def search_document(query, role="standard"):
    if not st.session_state.pdf_processed or st.session_state.vector_db is None:
        return "Please upload a PDF document first."
    
    # Initialize OpenAIClient if not already done
    if "openai_client" not in st.session_state:
        st.session_state.openai_client = OpenAIClient()
    
    # Check if this is a follow-up question
    is_follow_up = st.session_state.conversation_handler.detect_follow_up_question(query)
    
    # Search the vector database for relevant chunks
    results = st.session_state.vector_db.search(query, n_results=3)
    
    if not results['documents'] or not results['documents'][0]:
        return "No relevant information found in the document."
    
    # Format retrieved chunks for context
    context = format_retrieved_context(results)
    
    # Get the appropriate prompts based on the selected role
    system_prompt = get_system_prompt(role)
    
    # Use different prompts for follow-up questions
    if is_follow_up:
        user_prompt = st.session_state.conversation_handler.format_conversational_prompt(query, context)
    else:
        user_prompt = format_user_prompt(query, context, role)
    
    # Generate response using OpenAI
    try:
        answer = st.session_state.openai_client.get_response(
            prompt=user_prompt,
            system_prompt=system_prompt
        )
        
        # Format the answer part only (without sources)
        answer_part = f"**Answer:**\n{answer}\n\n"
        
        # Format the sources part
        sources_part = ""
        for i, (doc, distance) in enumerate(zip(
            results['documents'][0],
            results['distances'][0]
        )):
            sources_part += f"**Excerpt {i+1}** (Relevance: {100 - int(distance * 100)}%):\n"
            sources_part += f"{doc}\n\n"
        
        # Add to conversation history - add only the answer part
        st.session_state.conversation_handler.add_exchange(
            user_query=query,
            assistant_response=answer,
            context_used=context
        )
        
        # Return both parts separately for rendering
        return {
            "answer": answer_part,
            "sources": sources_part
        }
        
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        return {
            "answer": error_msg,
            "sources": f"Here are the relevant excerpts:\n\n{context}"
        }
    
# Function to handle role selection
def on_role_change():
    selected_role = st.session_state.role_selector
    st.session_state.selected_role = selected_role

# App title and description
st.title("📚 PDF Q&A Assistant")
st.markdown("""
Upload a PDF document and ask questions about its content. 
The app will search for the most relevant sections and provide answers based on the document.
""")

# Sidebar for PDF upload and settings
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process PDF"):
            success, message = process_uploaded_pdf(uploaded_file)
            if success:
                st.success(message)
            else:
                st.error(message)
    
    if st.session_state.pdf_processed:
        st.success(f"Currently using: {st.session_state.pdf_name}")
        
        # Option to clear the current PDF
        if st.button("Clear Current PDF"):
            if st.session_state.vector_db:
                try:
                    st.session_state.vector_db.delete_collection()
                except:
                    pass
            
            # Reset session state
            st.session_state.vector_db = None
            st.session_state.pdf_processed = False
            st.session_state.pdf_name = None
            st.session_state.chat_history = []
            st.session_state.conversation_handler = ConversationHandler()
            st.rerun()

    # Add a divider
    st.markdown("---")
    
    # Add role selection dropdown
    st.header("Response Settings")
    
    # Define available roles with friendly display names
    role_options = {
        "standard": "Standard Assistant",
        "corporate_lawyer": "Corporate Lawyer",
        "economist": "Economist", 
        "critical_journalist": "Critical Journalist",
        "theologian": "Theologian"
    }
    
    # Create the dropdown with the current selected role as default
    st.selectbox(
        "Select response perspective:",
        options=list(role_options.keys()),
        format_func=lambda x: role_options[x],
        key="role_selector",
        on_change=on_role_change,
        index=list(role_options.keys()).index(st.session_state.selected_role)
    )
    
    st.caption("Choose a perspective to receive answers from different viewpoints.")
    
    # Conversation management options
    if st.session_state.chat_history:
        st.markdown("---")
        st.header("Conversation")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Conversation"):
                st.session_state.chat_history = []
                # Also clear the conversation handler
                st.session_state.conversation_handler = ConversationHandler()
                st.rerun()
        
        with col2:
            if st.button("Save Conversation"):
                # Generate filename with timestamp
                import time
                filename = f"conversation_{int(time.time())}.json"
                
                # Convert conversation to JSON
                import json
                conversation_data = {
                    "history": st.session_state.conversation_handler.history,
                    "chat_display": st.session_state.chat_history
                }
                
                # Save the file
                with open(filename, "w") as f:
                    json.dump(conversation_data, f, indent=2)
                
                st.success(f"Conversation saved to {filename}")

# Main chat interface
st.header("Ask about the document")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        # If this is a full message with answer and sources
        if isinstance(message.get("content"), dict) and "answer" in message["content"] and "sources" in message["content"]:
            # Display answer
            st.markdown(message["content"]["answer"])
            
            # Display sources in a collapsible section
            with st.expander("View Sources", expanded=False):
                st.markdown(message["content"]["sources"])
        else:
            # Display regular message content
            st.markdown(message["content"])

# Input for new questions
if prompt := st.chat_input("Ask a question about the document..."):
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display response
    with st.chat_message("assistant"):
        if not st.session_state.pdf_processed:
            response = "Please upload and process a PDF document first."
            st.markdown(response)
            
            # Add simple string response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        else:
            with st.spinner("Searching document..."):
                # Check if it's a follow-up question
                is_follow_up = st.session_state.conversation_handler.detect_follow_up_question(prompt)
                
                # Get the response (now as a dict with answer and sources)
                response = search_document(prompt, st.session_state.selected_role)
                
                # Display the answer part
                st.markdown(response["answer"])
                
                # For follow-up questions, don't show sources
                if is_follow_up:
                    st.caption("Follow-up question - sources hidden")
                    # Still keep the sources in the response object, but don't display them
                else:
                    # Display the sources in a collapsible section for non-follow-up questions
                    with st.expander("View Sources", expanded=False):
                        st.markdown(f"<div class='source-content'>{response['sources']}</div>", unsafe_allow_html=True)
            
            # Add the structured response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})

# Information area at the bottom
st.markdown("---")
st.markdown("""
**How it works:**
1. Upload a PDF document using the sidebar
2. Click "Process PDF" to analyze the document
3. Ask questions about the content in the chat
4. The app will search for relevant information and provide answers
""")