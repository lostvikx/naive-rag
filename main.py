from pathlib import Path

from modules.scrape import extract_text
from modules.chunks import create_chunks
from modules.embedding import create_embeddings, store_embeddings
from modules.retrieve import retrieve_chunks
from modules.llm_gemini import prompt_llm

def main():
    pdf_file = "assets/docs/mlp_doc.pdf"
    chroma_dir = "data/chroma_db"

    Path("data").mkdir(parents=True, exist_ok=True)
    Path(chroma_dir).mkdir(parents=True, exist_ok=True)

    pages = extract_text(pdf_file)
    chunks = create_chunks(pages)
    embeddings = create_embeddings(chunks)

    count = store_embeddings(embeddings, chunks, data_dir=chroma_dir)
    print(f"Complete: Stored {count} embeddings in {chroma_dir}")

    query = input("Enter Prompt: ").strip()
    matched_chunks = retrieve_chunks(query, data_dir=chroma_dir, k=5)

    response = prompt_llm(query, matched_chunks)
    print(f"Answer: {response}")


if __name__ == "__main__":
    main()
