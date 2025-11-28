🚀 Vector + Graph Native AI Retrieval Engine

A lightweight, model-free hybrid search system built for the Devforge challenge.

🔎 What This Project Is

A fully local Vector + Graph native database that supports:

Vector Search (hash-based embeddings, cosine-like similarity)

Graph Search (typed, weighted edges, BFS with depth limit)

Hybrid Search (weighted merge of vector + graph scores)

Full CRUD API for nodes, edges, and embeddings

SQLite persistence and FastAPI server

No external ML models. No FAISS. No cloud dependencies.

🧱 Architecture Overview
 ┌─────────────────────────┐
 │       FastAPI API       │  <-- /nodes, /edges, /search/*
 └─────────────┬───────────┘
               │
 ┌─────────────▼────────────┐
 │     HybridSearchService   │  <-- weighted merge
 │  final = v_w * vec + g_w * graph
 └──────────┬───────┬────────┘
            │       │
   ┌────────▼──┐ ┌──▼──────────┐
   │ VectorSvc │ │ GraphSvc     │
   │ dot-prod  │ │ BFS + weights│
   └───────────┘ └──────────────┘
            │       │
   ┌────────▼───────▼──────────┐
   │     DatabaseManager        │
   │  SQLite: nodes/edges/emb   │
   └────────────────────────────┘

Core Modules

EmbeddingService – 256-dim hash-based embedding (deterministic)

VectorSearchService – full-scan vector similarity

GraphService – BFS traversal + hop/weight scoring

HybridSearchService – combines both signals

DatabaseManager – persistent store for nodes, embeddings, edges

⚡ Features
✔ Hash-based Embeddings

No ML models. Fast, deterministic, fully local.

✔ Weighted Graph Traversal

Supports:

Directed edges

Typed relationships

Weighted paths

Depth-limited BFS

✔ Hybrid Retrieval
final_score = vector_weight * vector_score
             + graph_weight  * graph_score

✔ CRUD for Nodes & Edges

Including embedding regeneration and cascading deletes.

✔ Full Automated Test Suite

Run all tests with:

python3 test_final.py


Covers API, CRUD, vector search, graph traversal, and hybrid correctness.

🛠 Installation
git clone <repo-url>
cd <repo>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Start the Server
uvicorn app:app --reload --host 127.0.0.1 --port 8000


Visit the interactive API docs at:

👉 http://127.0.0.1:8000/docs

🔧 Example Usage
Create a Node
POST /nodes
{
  "text": "Deep learning overview",
  "metadata": { "type": "note" }
}

Create an Edge
POST /edges
{
  "source": "node-A",
  "target": "node-B",
  "type": "cites",
  "weight": 1.0
}

Vector Search
POST /search/vector
{
  "query_text": "deep learning",
  "top_k": 5
}

Graph Search
GET /search/graph?start_id=node-A&depth=2&type=cites

Hybrid Search
POST /search/hybrid
{
  "query_text": "deep learning",
  "vector_weight": 0.7,
  "graph_weight": 0.3,
  "graph_start_id": "node-A",
  "graph_depth": 2,
  "top_k": 10
}

🧪 Test Suite

test_final.py validates:

API & CRUD

Node create/read/update/delete

Edge lifecycle & cascade deletion

Vector Search

Cosine similarity ordering

top_k > dataset size

Metadata filtering

Graph Traversal

BFS depth limiting

Typed relationship filtering

Cycle handling

Hybrid Search

Weighted merge correctness

Vector-only vs graph-only extremes

Run everything:

python3 test_final.py

📌 Notes

Embeddings are not semantic; they are deterministic hashed vectors.

Vector search is full-scan (simple & transparent).

Graph scoring is deterministic and interpretable.

Hybrid search is intentionally simple for clarity and reproducibility.

🎯 Summary

This repository implements a complete vector + graph native retrieval engine with:

Deterministic local embeddings

Weighted BFS graph scoring

Hybrid ranking

Full CRUD API

Automated evaluation script

Fast, local, interpretable — and tailor-made for the Devforge challenge.
