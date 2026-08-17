from retrieve import retrieve_chunks
from openai import OpenAI
import os
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# Continuous question-answer loop
while True:

    query = input("\nEnter your query (type 'exit' to quit): ")

    if query.lower() == "exit":
        print("Exiting...")
        break

    # Retrieve top 3 relevant chunks
    results = retrieve_chunks(query)

    # Combine retrieved chunks into context
    context = "\n\n".join(
        chunk for chunk, score in results
    )

    # Create RAG prompt
    prompt = f"""
Answer the question using only the context provided below.

Context:
{context}

Question:
{query}

Answer:
"""

    # Display prompt
    print("\n===== RAG PROMPT =====")
    print(prompt)

    # Send prompt to LLM
    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt
    )

    # Display answer
    print("\n===== ANSWER =====")
    print(response.output_text)