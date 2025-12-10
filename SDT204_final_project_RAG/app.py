import os
import argparse
import json
import chromadb
import ollama
from pypdf import PdfReader

CHROMA_PATH = "./chroma_db"
HISTORY_FILE = "./chat_history.json"
DATA_PATH = "./data"

LLM_MODEL = "mistral"
EMBED_MODEL = "nomic-embed-text"

ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
print(f"--- Connecting to Ollama at: {ollama_host} ---")

ollama_client = ollama.Client(host=ollama_host)


def get_embedding(text):
    clean_text = text.strip()
    if not clean_text:
        return None
    try:
        response = ollama_client.embeddings(model=EMBED_MODEL, prompt=clean_text)
        return response["embedding"]
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    name="rag_collection",
    metadata={"hnsw:space": "cosine"}
)


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return {}
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content: return {}
            return json.loads(content)
    except:
        return {}


def save_history(history_data):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history_data, f, indent=4, ensure_ascii=False)


def extract_text_from_file(filepath):
    text = ""
    try:
        if filepath.endswith('.pdf'):
            reader = PdfReader(filepath)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        elif filepath.endswith('.txt'):
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
    except Exception as error:
        print(f"Error reading file {filepath}: {error}")
    return text


def chunk_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def index_files(folder_path):
    print(f"--- Scanning {folder_path} ---")
    if not os.path.exists(folder_path):
        print(f"Directory {folder_path} not found.")
        return

    files_processed = 0
    chunks_added = 0
    chunks_skipped = 0

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.txt') or file.endswith('.pdf'):
                file_path = os.path.join(root, file)

                content = extract_text_from_file(file_path)
                if not content.strip():
                    continue

                chunks = chunk_text(content)

                ids = []
                documents = []
                metadatas = []

                for i, chunk in enumerate(chunks):
                    # Stable ID for deduplication
                    file_id = f"{file}_{i}"
                    ids.append(file_id)
                    documents.append(chunk)
                    metadatas.append({"source": file, "chunk_index": i})

                if not ids: continue

                existing_records = collection.get(ids=ids)
                existing_ids = set(existing_records['ids'])

                new_ids = []
                new_docs = []
                new_metas = []

                for idx, doc_id in enumerate(ids):
                    if doc_id not in existing_ids:
                        new_ids.append(doc_id)
                        new_docs.append(documents[idx])
                        new_metas.append(metadatas[idx])
                    else:
                        chunks_skipped += 1

                if new_ids:
                    print(f"Indexing {file}: Adding {len(new_ids)} chunks...")

                    new_embeddings = []
                    valid_indices = []

                    for k, doc_text in enumerate(new_docs):
                        emb = get_embedding(doc_text)
                        if emb:
                            new_embeddings.append(emb)
                            valid_indices.append(k)

                    final_ids = [new_ids[k] for k in valid_indices]
                    final_docs = [new_docs[k] for k in valid_indices]
                    final_metas = [new_metas[k] for k in valid_indices]

                    if final_ids:
                        collection.add(
                            ids=final_ids,
                            documents=final_docs,
                            embeddings=new_embeddings,  # Pass manually
                            metadatas=final_metas
                        )
                        chunks_added += len(final_ids)

                files_processed += 1

    print(f"\nSummary:")
    print(f"Files Scanned: {files_processed}")
    print(f"Chunks Added:  {chunks_added}")
    print(f"Chunks Skipped: {chunks_skipped} (Duplicates)")


def query_llm(question, context):
    formatted_prompt = f"""
    You are a helpful assistant. Use the provided context to answer the question.
    Be very attentive. Give concise, specific response, unless user asks some general questions
    If the answer is not in the context, say "I don't know based on the context.
    "

    Context: {context}

    Question: {question}
    """

    print("Thinking...")
    response = ollama_client.chat(model=LLM_MODEL, messages=[
        {'role': 'user', 'content': formatted_prompt}
    ])
    return response['message']['content']


def ask_question(question, chat_history_key="single_questions"):
    query_vec = get_embedding(question)
    if not query_vec:
        print("Error: Could not embed question (Ollama connection issue?).")
        return

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=5
    )

    if not results['documents'] or not results['documents'][0]:
        print("No relevant context found.")
        return

    retrieved_docs = results['documents'][0]
    sources = [m['source'] for m in results['metadatas'][0]]

    print(f"\n[Context Sources]: {list(set(sources))}")

    context_text = "\n\n".join(retrieved_docs)

    answer = query_llm(question, context_text)

    print("\nAnswer:")
    print(answer)

    hist = load_history()
    if chat_history_key not in hist:
        hist[chat_history_key] = []

    hist[chat_history_key].append({
        "question": question,
        "answer": answer,
        "sources": sources
    })
    save_history(hist)
    return answer


def chat_mode(chat_name, continue_chat=False):
    hist = load_history()

    if chat_name not in hist:
        if continue_chat:
            print(f"Chat '{chat_name}' not found.")
            return
        hist[chat_name] = []
        print(f"Starting chat: '{chat_name}'")
    else:
        print(f"Resuming chat: '{chat_name}'")
        for msg in hist[chat_name][:]:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "user": print(f"You: {content}")
            if role == "assistant": print(f"AI: {content}")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        query_vec = get_embedding(user_input)
        if not query_vec: continue

        results = collection.query(query_embeddings=[query_vec], n_results=3)
        context_docs = results['documents'][0] if results['documents'] else []
        context_text = "\n\n".join(context_docs)

        messages = []

        system_prompt = f"Use this context to help answer: {context_text}"
        messages.append({'role': 'system', 'content': system_prompt})

        raw_history = hist[chat_name][-6:]  # Last 6 messages
        for h in raw_history:
            if 'role' in h and 'content' in h:
                messages.append({'role': h['role'], 'content': h['content']})

        messages.append({'role': 'user', 'content': user_input})

        print("Thinking...")

        response = ollama_client.chat(model=LLM_MODEL, messages=messages)
        answer = response['message']['content']

        print(f"Assistant: {answer}")

        hist[chat_name].append({"role": "user", "content": user_input})
        hist[chat_name].append({"role": "assistant", "content": answer})
        save_history(hist)


def main():
    parser = argparse.ArgumentParser(description="Pure Python Local RAG")
    subparsers = parser.add_subparsers(dest='command')

    idx_parser = subparsers.add_parser('index')
    idx_parser.add_argument('--path', type=str, default=DATA_PATH)
    idx_parser.add_argument('--reset', action='store_true')

    ask_parser = subparsers.add_parser('ask')
    ask_parser.add_argument('question', type=str)

    chat_parser = subparsers.add_parser('chat')
    chat_parser.add_argument('name', type=str)

    sel_parser = subparsers.add_parser('select')
    sel_parser.add_argument('name', type=str)

    args = parser.parse_args()

    if args.command == 'index':
        if args.reset:
            print("Resetting database...")
            client.delete_collection("rag_collection")
            global collection
            collection = client.get_or_create_collection(
                name="rag_collection",
                metadata={"hnsw:space": "cosine"}
            )
        index_files(args.path)
    elif args.command == 'ask':
        ask_question(args.question)
    elif args.command == 'chat':
        chat_mode(args.name, continue_chat=False)
    elif args.command == 'select':
        chat_mode(args.name, continue_chat=True)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()