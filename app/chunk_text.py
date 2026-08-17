from extract_text import extract_text_from_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


pdf_path = "documents/attention_is_all_you_need.pdf"

text = extract_text_from_pdf(pdf_path)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)


chunks = text_splitter.split_text(text)


print("Total chunks:", len(chunks))

print("\n--- CHUNK 1 ---")
print(chunks[0])

print("\n--- CHUNK 2 ---")
print(chunks[1])

print("\n--- CHUNK LENGTHS ---")

for i, chunk in enumerate(chunks[:10]):
    print(f"Chunk {i+1}: {len(chunk)} characters")


print("\n--- CHUNK 3 ---")
print(chunks[2])

print("\n--- CHUNK 4 ---")
print(chunks[3])