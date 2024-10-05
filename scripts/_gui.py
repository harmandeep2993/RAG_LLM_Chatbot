import streamlit as st
from _4_query_handler import load_faiss_index, load_chunks_and_metadata, retrieve_top_k_chunks, generate_response, filter_relevant_chunks
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import _0_config
# Load the sentence-transformers embedding model
embedder = SentenceTransformer(_0_config.EMBEDDING_MODEL)

# Load the FAISS index and combined question-answer chunks only once
index = load_faiss_index()
combined_chunks = load_chunks_and_metadata()

# Define a confidence threshold to filter irrelevant queries
CONFIDENCE_THRESHOLD = 45  # Adjust this to fine-tune strictness

def keyword_match(query, chunk):
    """
    Check if any keyword from the query appears in the chunk.
    """
    query_keywords = set(query.lower().split())  # Split query into words
    chunk_keywords = set(chunk.lower().split())  # Split chunk into words

    # Return True if any word in query matches any word in the chunk
    return bool(query_keywords.intersection(chunk_keywords))

def semantic_match(query_embedding, chunk_embeddings, threshold=0.7):
    """
    Matches the query to the chunk using cosine similarity between embeddings.
    Ensures the query_embedding is properly reshaped and chunk_embeddings is not empty.
    """
    if chunk_embeddings.size == 0:
        return []  # Return empty list if no chunks

    # Reshape query_embedding to ensure it's 2D (1 sample, multiple features)
    query_embedding = query_embedding.reshape(1, -1)

    # Calculate cosine similarities
    similarities = cosine_similarity(query_embedding, chunk_embeddings)
    
    # Filter chunks based on similarity threshold
    relevant_indices = [i for i, score in enumerate(similarities[0]) if score > threshold]
    
    return relevant_indices

def get_helpbee_response(query):
    # Step 1: Retrieve top chunks and distances
    top_chunks = retrieve_top_k_chunks(query, index, combined_chunks, k=3)

    # Step 2: Filter relevant chunks based on expanded keyword matching
    relevant_chunks = filter_relevant_chunks(query, top_chunks)

    # If no chunks pass keyword filtering, fallback to semantic matching
    if not relevant_chunks:
        query_embedding = embedder.encode([query], convert_to_numpy=True)
        
        # Ensure that there are chunks to compare against
        if top_chunks:
            chunk_embeddings = embedder.encode(top_chunks, convert_to_numpy=True)
            relevant_indices = semantic_match(query_embedding, chunk_embeddings)

            # Retrieve relevant chunks based on semantic similarity
            relevant_chunks = [top_chunks[i] for i in relevant_indices] if relevant_indices else []

    # Step 3: Generate a response
    if relevant_chunks:
        response = generate_response(query, relevant_chunks)
    else:
        response = "Sorry, I don't have information about that. Please ask another question."

    return response

def process_input():
    """
    Processes the user input and generates a response by calling the query handler.
    """
    user_query = st.session_state.user_input
    if user_query:
        st.session_state.conversation_history.append(f"You: {user_query}")
        st.session_state.conversation_history.append("Helpbee: I am calculating the response...")

        st.session_state.user_input = ""  # Clear the input

        # Get the response from Helpbee
        helpbee_response = get_helpbee_response(user_query)

        # Update the last response
        st.session_state.conversation_history[-1] = f"Helpbee: {helpbee_response}"

# Define function to handle clickable FAQ queries
def handle_click(query):
    """
    Handles the click event when a user clicks on a predefined FAQ query.
    
    Args:
        query (str): The clicked query text.
    """
    st.session_state.user_input = query  # Set the clicked FAQ as the user's input
    process_input()  # Process the input to get Helpbee's response

def chatbot_gui():
    """
    Set up the chatbot GUI using Streamlit.
    """
    st.set_page_config(layout="wide")

    # Add custom CSS for chat bubbles, smaller font in the left column, and a border divider
    st.markdown(
        '''
        <style>
        .user_message {
            background-color: #dee2e6;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
            color: #000000;
            text-align: right;
        }
        .helpbee_message {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
            color: #000000;
            text-align: left;
        }
        .small-text {
            font-size: 10px;
            text-align: justify;
            word-spacing: 2px;
        }
        .header {
            font-size: 40px;
            font-weight: bold;
            text-align: Left;
        }
        .header-text {
            font-size: 20px;
            font-weight: bold;
            text-align: left;
        }
        .subheader-text {
            font-size: 16px;
            font-weight: bold;
        }
        .padding {
            padding-left: 5px;  /* Add padding between the columns */
        }
        /* Border divider between left and right columns */
        .border-divider {
            border-left: 2px solid #D3D3D3; /* Light gray border */
            height: 100%;
        }
        div.stButton > button {
            font-size: 5px !important;
            color: black !important; /* Set text color to black */
            background-color: #dee2e6 !important; /* Set background color to #dee2e6 */
            border: none !important;
            text-decoration: none !important;
        }
        </style>
     ''', unsafe_allow_html=True
    )

    # Set up a two-column layout
    left_column, space_column, right_column = st.columns([1, 0.3, 2])

    with left_column:
        # Sidebar with FAQ
        st.markdown('<div class="header">Helpbee 🤖</div>', unsafe_allow_html=True)
        st.markdown('<div class="header-text">Your 24*7 Virtual Assistant</div>', unsafe_allow_html=True)
        st.markdown("")

        st.markdown('<div class="subheader-text">Frequently Asked Questions</div>', unsafe_allow_html=True)
        st.markdown('<div class="small-text">', unsafe_allow_html=True)
        
        # FAQ buttons that trigger the chatbot
        if st.button("How do I create an account?", key="q1"):
            handle_click("How do I create an account?")
        if st.button("How can I track my order?", key="q3"):
            handle_click("How can I track my order?")
        if st.button("What payment methods are accepted?", key="q4"):
            handle_click("What payment methods are accepted?")
        if st.button("How can I cancel my order?", key="q5"):
            handle_click("How can I cancel my order?")

        st.markdown('</div>', unsafe_allow_html=True)

    # Add a vertical divider
    with space_column:
        st.markdown('<div class="border-divider"></div>', unsafe_allow_html=True)

    # Chat window for conversation history and user input
    with right_column:
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = ["Helpbee: Hello! How can I assist you today?"]

        if "user_input" not in st.session_state:
            st.session_state.user_input = ""

        # Display conversation history
        chat_placeholder = st.empty()
        with chat_placeholder.container():
            for message in st.session_state.conversation_history:
                if message.startswith("You"):
                    st.markdown(f"<div class='user_message'>{message}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='helpbee_message'>{message}</div>", unsafe_allow_html=True)

        # Input box for user query
        user_input_placeholder = st.empty()
        user_input_placeholder.text_input("Type your message here:", key="user_input", on_change=process_input)

# Run the Streamlit app
if __name__ == "__main__":
    chatbot_gui()
