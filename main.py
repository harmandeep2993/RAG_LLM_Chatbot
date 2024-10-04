# main.py

# Import the Streamlit-based GUI module to launch the app
import os

# If Streamlit is already installed and `gui.py` is functioning correctly, this script will run the Streamlit app.

def main():
    print("Launching Helpbee Chatbot GUI...")
    os.system("streamlit run scripts/_gui.py")

if __name__ == "__main__":
    main()