import os
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# =========================
# CONFIG
# =========================
CSV_FILE = r"D:\Formula1_2025Season_RaceResults.csv"
DB_LOCATION = "./chroma_f1_db"
COLLECTION_NAME = "f1_race_results_2025"
BATCH_SIZE = 420

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_FILE)

# =========================
# EMBEDDING MODEL
# =========================
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# =========================
# VECTOR STORE
# =========================
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=DB_LOCATION,
    embedding_function=embeddings
)

existing_count = vector_store._collection.count()
print("Existing documents in DB:", existing_count)

# =========================
# PREPARE DOCUMENTS (MATCHES CSV)
# =========================
documents = []
ids = []

for i, row in df.iterrows():
    content = (
        f"At the {row.get('Track')} Grand Prix, "
        f"{row.get('Driver')} from {row.get('Team')} "
        f"finished position {row.get('Position')}. "
        f"They started from grid position {row.get('Starting Grid')}, "
        f"completed {row.get('Laps')} laps, "
        f"and finished with status/time '{row.get('Time/Retired')}'. "
        f"They scored {row.get('Points')} points. "
        f"Fastest lap: {row.get('Set Fastest Lap')} "
        f"with a time of {row.get('Fastest Lap Time')}."
    )

    doc = Document(
        page_content=content,
        metadata={
            "track": row.get("Track"),
            "driver": row.get("Driver"),
            "team": row.get("Team"),
            "position": row.get("Position"),
            "points": row.get("Points")
        },
        id=str(i)
    )

    documents.append(doc)
    ids.append(str(i))

# =========================
# INGEST
# =========================
if existing_count == 0:
    print("Ingesting F1 2025 race results into Chroma...")

    for i in range(0, len(documents), BATCH_SIZE):
        batch_docs = documents[i:i + BATCH_SIZE]
        batch_ids = ids[i:i + BATCH_SIZE]

        vector_store.add_documents(
            documents=batch_docs,
            ids=batch_ids
        )

        print(f"Inserted documents {i} to {i + len(batch_docs)}")

    print("Final document count:", vector_store._collection.count())
else:
    print("Using existing embeddings. No re-ingestion needed.")

# =========================
# RETRIEVER
# =========================
retriever = vector_store.as_retriever(
    search_kwargs={"k": 15}
)
