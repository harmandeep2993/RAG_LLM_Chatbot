import streamlit as st
from _4_query_handler import load_faiss_index, load_chunks, retrieve_top_k_chunks, generate_response, filter_relevant_chunks

# Load the FAISS index and chunks only once
index = load_faiss_index()
text_chunks = load_chunks()

# Define a confidence threshold to filter irrelevant queries
CONFIDENCE_THRESHOLD = 45  # Lower the threshold to allow more chunks through

def keyword_match(query, chunk):
    """
    Check if any keyword from the query appears in the chunk.
    """
    query_keywords = set(query.lower().split())  # Split query into words
    chunk_keywords = set(chunk.lower().split())  # Split chunk into words

    # Return True if any word in query matches any word in the chunk
    return bool(query_keywords.intersection(chunk_keywords))

def get_helpbee_response(query):
    """
    Get the Helpbee chatbot response for the user query by interacting with the query handler.
    Args:
        query (str): The user's query.
    Returns:
        str: Helpbee's response or a fallback message if the query is out of scope.
    """
    # Step 1: Retrieve top chunks and distances
    top_chunks, distances = retrieve_top_k_chunks(query, index, text_chunks, k=3, return_distances=True)

    # Step 2: Check if the best match is within the confidence threshold
    if distances[0] < CONFIDENCE_THRESHOLD:
        relevant_chunks = filter_relevant_chunks(query, top_chunks)

        # Step 3: If there are relevant chunks, generate a response
        if relevant_chunks:
            response = generate_response(query, relevant_chunks)
        else:
            # No relevant chunks found, return fallback response
            response = "Sorry, I don't have information about that. Please ask another question."
    else:
        # Distance too high, meaning no good match in the knowledge base
        response = "Sorry, I couldn't find any relevant information for your query. Please try rephrasing or asking a different question."

    return response

def process_input():
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
    # Set up Streamlit page with a two-column layout
    st.set_page_config(layout="wide")

    # Add some CSS for chat bubbles, smaller font in the left column, and a border divider
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
            color: black !important; /* Set text color to white */
            background-color: #dee2e6 !important; /* Set background color to #dee2e6 */
            border: none !important;
            text-decoration: none !important;
        }
        </style>
     ''', unsafe_allow_html=True
    )

    # Adjust the column layout to include space between the left and right columns with a border divider
    left_column, space_column, right_column = st.columns([1, 0.3, 2])  # Narrow space column for the border

    with left_column:
        # Change header and subheader sizes
        st.markdown('<div class="header">Helpbee 🤖</div>', unsafe_allow_html=True)
        st.markdown('<div class="header-text">Your 24*7 Virtual Assistant</div>', unsafe_allow_html=True)
        st.markdown("")
        st.markdown('<div class="subheader-text">About me</div>', unsafe_allow_html=True)
        st.markdown('<div class="small-text">', unsafe_allow_html=True)
        st.write("""
            Hey there 👋 , I'm Helpbee, your 24/7 shopping sidekick! 🐝 Think of me as the hyperactive, caffeine-fueled bee that's here to make your shopping experience as smooth as honey.

        Here’s what I can do for you:
        """)
        # Display each button inside its own blue box
        if st.button("How do I create an account?", key="q1"):
            handle_click("How do I create an account?")
        if st.button("How can I track my order?", key="q3"):
            handle_click("How can I track my order?")
        if st.button("What payment methods are accepted?", key="q4"):
            handle_click("What payment methods are accepted?")
        if st.button("How can I cancel my order?", key="q5"):
            handle_click("How can I cancel my order?")
            st.markdown('</div>', unsafe_allow_html=True)
    # Add border divider between columns
    with space_column:
        st.markdown('<div class="border-divider"></div>', unsafe_allow_html=True)

    # Right column for chat interface
    with right_column:
        # Add some vertical spacing in the right column to match the height of the left column's heading
        st.write("")  # Adding an empty element to create vertical space
        st.write("")  # Adjust as necessary, add more empty st.write() calls or use st.empty() for more spacing
        st.write("") 
        st.write("")
        st.write("") 
        st.write("")
        # Initialize conversation history in session state
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = ["Helpbee: Hello! How can I assist you today?"]

        if "user_input" not in st.session_state:
            st.session_state.user_input = ""

        # Chat history display area
        chat_placeholder = st.empty()

        with chat_placeholder.container():
            for message in st.session_state.conversation_history:
                if message.startswith("You"):
                    st.markdown(f"<div class='user_message'>{message}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='helpbee_message'>{message}</div>", unsafe_allow_html=True)

        # Input box at the bottom for user to type a message
        user_input_placeholder = st.empty()
        user_input_placeholder.text_input("Type your message here:", key="user_input", on_change=process_input)

# Run the Streamlit app
if __name__ == "__main__":
    chatbot_gui()



# import streamlit as st
# from _4_query_handler import load_faiss_index, load_chunks, retrieve_top_k_chunks, generate_response, filter_relevant_chunks

# # Load the FAISS index and chunks only once
# index = load_faiss_index()
# text_chunks = load_chunks()

# # Define a confidence threshold to filter irrelevant queries
# CONFIDENCE_THRESHOLD = 45  # Adjust this to fine-tune strictness

# def get_helpbee_response(query):
#     """
#     Get the Helpbee chatbot response for the user query by interacting with the query handler.
    
#     Args:
#         query (str): The user's query.
        
#     Returns:
#         str: Helpbee's response or a fallback message if the query is out of scope.
#     """
#     # Retrieve the top 3 most relevant chunks and their distances
#     top_chunks, distances = retrieve_top_k_chunks(query, index, text_chunks, k=3, return_distances=True)
    
#     # Print distances for debugging purposes
#     print(f"Query: {query}")
#     print(f"Distances: {distances}")

#     # Check if the best match is within the confidence threshold
#     if distances[0] < CONFIDENCE_THRESHOLD:
#         # Filter relevant chunks to make sure they are related to the query
#         relevant_chunks = filter_relevant_chunks(query, top_chunks)
        
#         # Extra step: Ensure the response contains relevant keywords from the query
#         if any(keyword in relevant_chunks[0].lower() for keyword in query.lower().split()):
#             # Generate the final response using the FLAN-T5 model
#             response = generate_response(query, relevant_chunks)
#         else:
#             # If keywords don't match, return a fallback message
#             response = "Sorry, I couldn't find a precise answer for your query."
#     else:
#         # If the distance is too high, return a fallback message
#         response = "Sorry, I can only answer questions related to our FAQ."
    
#     return response

# def process_input():
#     # Process input only if it's not empty
#     user_query = st.session_state.user_input
#     if user_query:
#         # Add the user's query to the conversation history instantly
#         st.session_state.conversation_history.append(f"You: {user_query}")
        
#         # Show a temporary response from Helpbee while calculating
#         st.session_state.conversation_history.append("Helpbee: I am calculating the response...")

#         # Clear the input field by resetting the session state for input
#         st.session_state.user_input = ""  # Reset the input field

#         # Calculate the Helpbee response
#         helpbee_response = get_helpbee_response(user_query)

#         # Remove the "calculating" message and add the actual response
#         st.session_state.conversation_history[-1] = f"Helpbee: {helpbee_response}"

# # Define the Streamlit GUI application
# def chatbot_gui():
#     # Set up Streamlit page
#     st.title("Helpbee 🤖! Your Virtual Assistant")
#     st.write("---")  # Separator line
    
#     # Initialize session state for conversation history and user input if not already initialized
#     if "conversation_history" not in st.session_state:
#         st.session_state.conversation_history = [
#             "Helpbee: Hello! How can I assist you today?"
#         ]  # Start with Helpbee's greeting

#     if "user_input" not in st.session_state:
#         st.session_state.user_input = ""  # Start with an empty input field

#     # Create a placeholder for the conversation history that we will update dynamically
#     history_placeholder = st.empty()

#     # Use the placeholder to display conversation history
#     with history_placeholder.container():
#         for message in st.session_state.conversation_history:
#             st.write(message)

#     # Input box to take user's question (auto-submits on pressing Enter)
#     st.text_input("Your message:", 
#                   value=st.session_state.user_input, 
#                   placeholder="Please type your message and press Enter", 
#                   key="user_input", 
#                   on_change=process_input)  # Triggers on Enter key press

# # Run the Streamlit app
# if __name__ == "__main__":
#     chatbot_gui()


# import streamlit as st
# from _4_query_handler import load_faiss_index, load_chunks, retrieve_top_k_chunks, generate_response, filter_relevant_chunks

# # Load the FAISS index and chunks only once
# index = load_faiss_index()
# text_chunks = load_chunks()

# def get_chatbot_response(query):
#     """
#     Get the chatbot response for the user query by interacting with the query handler.
    
#     Args:
#         query (str): The user's query.
        
#     Returns:
#         str: The chatbot's response.
#     """
#     # Retrieve the top 5 most relevant chunks
#     top_chunks = retrieve_top_k_chunks(query, index, text_chunks, k=5)
#     # Filter relevant chunks to make sure they are related to the query
#     relevant_chunks = filter_relevant_chunks(query, top_chunks)
#     # Generate the final response using the FLAN-T5 model
#     response = generate_response(query, relevant_chunks)
#     return response

# # Define the Streamlit GUI application
# def chatbot_gui():
#     # Set up Streamlit page
#     st.title("Helpbee 🤖! Your Virtual Assistant")
#     st.write("---")  # Separator line
    
#     # Initialize session state for conversation history and user input if not already initialized
#     if "conversation_history" not in st.session_state:
#         st.session_state.conversation_history = [
#             "Chatbot: Hello! How can I assist you today?"
#         ]  # Start with the bot's greeting

#     if "user_input" not in st.session_state:
#         st.session_state.user_input = ""  # Start with an empty input field

#     # Create a placeholder for the conversation history that we will update dynamically
#     history_placeholder = st.empty()

#     # Use the placeholder to display conversation history
#     with history_placeholder.container():
#         for message in st.session_state.conversation_history:
#             st.write(message)

#     # Place the input and submit button at the bottom of the page
#     user_query = st.text_input("Your message:", placeholder=" Please type your message?", key="input_box")
#     if st.button("Submit"):
#         if user_query:
#             # Add the user's query to the conversation history
#             st.session_state.conversation_history.append(f"You: {user_query}")
            
#             # Get the chatbot response
#             chatbot_response = get_chatbot_response(user_query)

#             # Check if chatbot response exists
#             if chatbot_response:
#                 # Append the chatbot's response to the conversation history
#                 st.session_state.conversation_history.append(f"Chatbot: {chatbot_response}")
#             else:
#                 # Handle case where no response was found (e.g., when query doesn't match)
#                 st.session_state.conversation_history.append(f"Chatbot: Sorry, I couldn't find an answer for that.")

#             # Clear the input box after submission by setting `st.session_state.user_input` to an empty string
#             st.session_state.user_input = ""

#             # Update the conversation history display after submission
#             with history_placeholder.container():
#                 for message in st.session_state.conversation_history:
#                     st.write(message)

# # Run the Streamlit app
# if __name__ == "__main__":
#     chatbot_gui()


# The normal GuiÖ

# import tkinter as tk
# from tkinter import scrolledtext
# from _4_query_handler import load_faiss_index, load_chunks, retrieve_top_k_chunks, generate_response, filter_relevant_chunks

# # Load the FAISS index and chunks only once
# index = load_faiss_index()
# text_chunks = load_chunks()

# def get_chatbot_response(query):
#     """
#     Get the chatbot response for the user query by interacting with the query handler.
    
#     Args:
#         query (str): The user's query.
        
#     Returns:
#         str: The chatbot's response.
#     """
#     top_chunks = retrieve_top_k_chunks(query, index, text_chunks, k=5)  # Retrieve top chunks
#     relevant_chunks = filter_relevant_chunks(query, top_chunks)  # Filter relevant chunks
#     response = generate_response(query, relevant_chunks)  # Generate response
#     return response

# # Define the GUI application
# def chatbot_gui():
#     # Create the main window
#     window = tk.Tk()
#     window.title("RAG Chatbot")
#     window.geometry("600x500")
    
#     # Create a label for the user query
#     query_label = tk.Label(window, text="Enter your query:")
#     query_label.pack()
    
#     # Create a text input box for the user to enter their query
#     query_entry = tk.Entry(window, width=80)
#     query_entry.pack(pady=10)
    
#     # Create a scrolled text widget to display the conversation history
#     conversation_display = scrolledtext.ScrolledText(window, height=20, width=80, state='disabled')
#     conversation_display.pack(pady=10)
    
#     def on_submit():
#         # Get the user's query
#         user_query = query_entry.get()
        
#         if user_query:
#             # Display the user's query in the chat window
#             conversation_display.configure(state='normal')  # Enable editing in the text area
#             conversation_display.insert(tk.END, f"You: {user_query}\n")
            
#             # Get the chatbot response
#             chatbot_response = get_chatbot_response(user_query)
            
#             # Display the chatbot's response in the chat window
#             conversation_display.insert(tk.END, f"Chatbot: {chatbot_response}\n\n")
            
#             # Scroll to the bottom of the conversation display
#             conversation_display.yview(tk.END)
            
#             # Disable the text area again to prevent manual input
#             conversation_display.configure(state='disabled')
            
#             # Clear the input field for the next question
#             query_entry.delete(0, tk.END)
#         else:
#             conversation_display.configure(state='normal')
#             conversation_display.insert(tk.END, "Please enter a query.\n")
#             conversation_display.configure(state='disabled')
    
#     # Create a submit button to get the response
#     submit_button = tk.Button(window, text="Submit", command=on_submit)
#     submit_button.pack(pady=10)
    
#     # Start the GUI event loop
#     window.mainloop()

# # Run the GUI
# if __name__ == "__main__":
#     chatbot_gui()