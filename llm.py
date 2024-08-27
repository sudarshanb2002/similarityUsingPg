import os
import numpy as np
import google.generativeai as genai
import psycopg2

# Directly define the API key for Gemini AI
gemini_api_key = "AIzaSyDTj1X3OuixdaIFpA4VMlxD8HUm9t8rngk"

# Configure the Gemini AI client
genai.configure(api_key=gemini_api_key)

# Create the model with specific configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",  # Replace with the appropriate model name
    generation_config=generation_config,
)

def get_embedding(text):
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document",
        title="Embedding of single string"
    )
    embedding = np.array(result['embedding']).astype(np.float32)
    return embedding


def connect_to_postgresql():
    """Connect to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname="postgres",
            user="postgres",
            password="postgres",
            host="172.22.82.158",
            port="5432"
        )
        print("Connected to PostgreSQL successfully.")
        return conn
    except Exception as e:
        print(f"Failed to connect to PostgreSQL: {e}")
        raise e

def split_text(text, chunk_size):
    """Split text into chunks of a specific size."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def truncate_text(text, max_length):
    """Truncate text to the maximum allowed length."""
    return text[:max_length]

def insert_chunks_to_postgresql(content):
    """Insert truncated content into multiple rows in PostgreSQL."""
    try:
        conn = connect_to_postgresql()
        cursor = conn.cursor()

        title = truncate_text(content["title"], 255)
        headings = truncate_text(", ".join(content["headings"]), 1000)
        paragraphs = " ".join(content["paragraphs"])
        lists = truncate_text(" ".join(content["lists"]), 1000)
        description = truncate_text(content.get("description", ""), 1000)

        # Split the data into chunks of 1000 characters
        paragraph_chunks = split_text(paragraphs, 1000)
        list_chunks = split_text(lists, 1000)

        # Insert each chunk into the PostgreSQL table
        for i, chunk in enumerate(paragraph_chunks):
            embedding_text = f"{title} {headings} {chunk}".strip()
            embedding = get_embedding(embedding_text).tolist()
            cursor.execute("""
                INSERT INTO documents (title, paragraphs, embedding)
                VALUES (%s, %s, %s);
            """, (title, chunk, embedding))
        
        for i, chunk in enumerate(list_chunks):
            embedding_text = f"{title} {headings} {chunk}".strip()
            embedding = get_embedding(embedding_text).tolist()
            cursor.execute("""
                INSERT INTO documents (title, lists, embedding)
                VALUES (%s, %s, %s);
            """, (title, chunk, embedding))
        
        conn.commit()
        print("Data inserted successfully into PostgreSQL.")
    except Exception as e:
        print(f"Failed to insert data into PostgreSQL: {e}")
        conn.rollback()
        raise e

def perform_similarity_search(query_text):
    """Perform similarity search using PostgreSQL and format results."""
    try:
        top_k=10
        conn=connect_to_postgresql()

        cursor = conn.cursor()

        # Generate the embedding for the query text
        query_embedding = get_embedding(query_text)
        # Generate the embedding for the query text
        query_embedding_array = np.array(query_embedding)

        # Convert the array to a list
        query_embedding_list = query_embedding_array.tolist()

        # Convert the list to a format that PostgreSQL expects
        query_embedding_vector = f"ARRAY{query_embedding_list}::vector"
        # print("query_embedding_vector--------------------", query_embedding_vector)

        # Use the correct SQL query with the embedding array
        query = f"""
          SELECT paragraphs
          FROM documents
          ORDER BY embedding <=> {query_embedding_vector}
          LIMIT %s;
        """
        cursor.execute(query, (1,))

        # Execute the SQL query
# Fetch and return the paragraphs
        results = cursor.fetchall()

        # Extracting just the paragraphs from the results
        paragraphs = [result[0] for result in results]

    
        # Generate a human-readable summary using the paragraphs
        summary = generate_human_readable_summary(query_text, paragraphs)

        return summary

    except Exception as e:
        print(f"Failed to perform similarity search: {e}")
        raise e

def generate_human_readable_summary(query_text, paragraphs):
    """Generate a human-readable summary of the results using the LLM."""
    chat_session = model.start_chat(history=[])

    # Define the prompt for the chat model
    prompt = f"""
        I performed a similarity search on a collection of documents using the query text "{query_text}". Below is a summary of the top results from the similarity search:

    Results:
    {paragraphs}

    Your task is to generate a coherent and relevant content piece based on the query text and the provided search results. The content should be aligned with the query and reflect the information in the retrieved paragraphs. Avoid summarizing the results; instead, create content that incorporates the key elements from the query and the search results.

    Query Text: {query_text}
    """

    # Send the prompt to the chat model
    response = chat_session.send_message(prompt)
    print("Generated Summary:", response.text)

    return response.text



# if __name__ == "__main__":
#     # Example mock data
#     mock_data = {
#         "title": "The History of Milvus",
#         "headings": [
#             "Introduction",
#             "Early History",
#             "Modern Usage",
#             "Technological Advances",
#             "Future Prospects"
#         ],
#         "paragraphs": [
#             "Milvus, a genus of birds, has a rich history spanning across various cultures...",
#             "In the early days, Milvus species were commonly found in the Eurasian and African regions...",
#             "In modern times, Milvus species have adapted to various habitats including urban areas...",
#             "Technological advances have allowed researchers to track the movements of Milvus in real-time...",
#             "Looking ahead, researchers are focusing on conservation efforts to protect Milvus species..."
#         ],
#         "lists": [
#             "Key Features of Milvus:",
#             "1. Sharp talons",
#             "2. Strong beak",
#         ],
#         "description": "Milvus, commonly known as kites, are a genus of birds that are known for their graceful flight..."
#     }

#     # Connect to PostgreSQL and insert data
#     conn = connect_to_postgresql()
#     # insert_to_postgresql(conn, mock_data)

#     # Perform a similarity search
#     query_text = "what is milvus?"
#     perform_similarity_search(conn, query_text)

#     # Close the connection
#     conn.close()
