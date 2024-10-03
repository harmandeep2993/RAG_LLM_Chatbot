import numpy as np
import time

# 1. Measure Response Time
def measure_response_time(query, chatbot_function):
    """
    Measures the time taken by the chatbot to respond.
    """
    start_time = time.time()  # Record start time
    response = chatbot_function(query)  # Get chatbot response
    end_time = time.time()  # Record end time
    
    response_time = end_time - start_time
    return response, response_time


# 2. Calculate Relevance
def calculate_relevance(query, response, keywords):
    """
    Calculates relevance of the chatbot response based on matching keywords from the query.
    """
    query_words = set(query.lower().split())
    response_words = set(response.lower().split())
    
    # Count keyword matches between query and response
    matches = len(query_words.intersection(response_words.intersection(set(keywords))))
    
    # Simple 3-point relevance scale
    if matches > 3:
        score = 3  # Fully relevant
    elif matches > 1:
        score = 2  # Somewhat relevant
    else:
        score = 1  # Not relevant
    
    return score


# 3. Calculate Confidence (from FAISS or retrieval)
def calculate_confidence(faiss_distances):
    """
    Converts FAISS distances into confidence scores (smaller distance = higher confidence).
    """
    normalized_confidences = (1 - np.array(faiss_distances)) * 100  # Normalize to 0-100 scale
    return normalized_confidences[0]  # Return confidence of the top match


# 4. Calculate Accuracy
def calculate_accuracy(response, ground_truth):
    """
    Compares the chatbot response with the ground truth to determine accuracy.
    """
    return 1 if response.lower().strip() == ground_truth.lower().strip() else 0


# 5. Evaluate Chatbot for a Single Query
def evaluate_single_query(query, ground_truth, chatbot_function, faiss_distances):
    """
    Evaluates the chatbot based on accuracy, relevance, response time, and confidence for a single query.
    """
    # 5.1 Measure response time and get chatbot response
    response, response_time = measure_response_time(query, chatbot_function)
    
    # 5.2 Calculate metrics
    accuracy = calculate_accuracy(response, ground_truth)
    relevance = calculate_relevance(query, response, ground_truth.split())  # Using keywords from ground truth
    confidence = calculate_confidence(faiss_distances)  # FAISS confidence score

    return accuracy, relevance, response_time, confidence, response


# Example Chatbot Function (Replace with actual chatbot function)
def chatbot_function(query):
    """
    This is a placeholder chatbot function. Replace it with the real chatbot function.
    """
    time.sleep(0.5)  # Simulate delay for response
    return "This is a placeholder response."


# Example: Taking User Input
query = input("Enter your query: ")  # Taking query from the customer
ground_truth = input("Enter the ideal response: ")  # Taking ideal response (ground truth) from the customer

# Example FAISS distances
faiss_distances = [0.3, 0.4, 0.5]  # Example FAISS distances (smaller = more confident retrieval)

# Run evaluation for the single query
accuracy, relevance, response_time, confidence, chatbot_response = evaluate_single_query(
    query, ground_truth, chatbot_function, faiss_distances
)

# Print the results
print(f"\nResults for Query: {query}")
print(f"Chatbot Response: {chatbot_response}")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Relevance: {relevance:.2f}")
print(f"Response Time: {response_time:.2f} seconds")
print(f"Confidence: {confidence:.2f}%")