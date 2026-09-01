import streamlit as st
import ollama

# Set up the Streamlit app
def main():
    # Set page layout
    st.set_page_config(page_title="Mistral Chatbot", layout="centered")
    
    # Title and instructions
    st.title("Mistral Chatbot 🤖")
    st.write("Ask any question, and the Mistral model will generate a response!")

    # Input box for user query
    user_input = st.text_input("Enter your question:", "")

    # Display response after the user submits a query
    if st.button("Get Response"):
        if user_input.strip():
            with st.spinner("Generating response..."):
                response = get_mistral_response(user_input)
                st.write(f"**Response:** {response}")
        else:
            st.error("Please enter a question before submitting.")

# Function to call the Mistral model via Ollama
def get_mistral_response(query):
    # Custom prompt format
    prompt = f"""
    You are Helpbee, an assistant designed to respond to customer questions. Based on the information below, answer the user's question clearly and concisely.

    Question: {query}
    """

    # Use the Ollama API to send the query to the Mistral model
    try:
        response = ollama.generate(model="mistral", prompt=prompt)
        return response['response']
    except Exception as e:
        return f"Error generating response: {e}"

# Run the app
if __name__ == "__main__":
    main()