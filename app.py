"""
Vector + Graph Native Database for Efficient AI Retrieval
Implements hybrid search using weighted vector + graph scoring.

This version:
- Uses a simple hash-based embedding (no sentence-transformers, no Hugging Face, no FAISS)
- Stores embeddings in SQLite as float32 vectors
- Implements full CRUD for nodes and edges
- Supports metadata filtering in vector search
- Supports type filtering and weight-aware scoring in graph search
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from collections import defaultdict, deque

# ============ Pydantic Models ============

class NodeCreate(BaseModel):
    text: str
    metadata: Dict = Field(default_factory=dict)
    auto_embed: bool = True
    # Allow user-provided embeddings for tests like "known embeddings"
    embedding: Optional[List[float]] = None


class NodeUpdate(BaseModel):
    text: Optional[str] = None
    metadata: Optional[Dict] = None
    # Match test name: regen_embedding
    regen_embedding: bool = False
    # Optional manual embedding override (stretch)
    embedding: Optional[List[float]] = None


class EdgeCreate(BaseModel):
    source: str
    target: str
    type: str = "default"
    weight: float = 1.0


class EdgeUpdate(BaseModel):
    type: Optional[str] = None
    weight: Optional[float] = None


class VectorSearchRequest(BaseModel):
    query_text: str
    top_k: int = 10
    # Optional metadata filter, e.g. {"type": "note"}
    metadata_filter: Optional[Dict[str, str]] = None


class HybridSearchRequest(BaseModel):
    query_text: str
    vector_weight: float = 0.5
    graph_weight: float = 0.5
    top_k: int = 10
    graph_start_id: Optional[str] = None
    graph_depth: int = 2


# ============ Database Manager ============

class DatabaseManager:
    def __init__(self, db_path: str = "graph_vector.db"):
        self.db_path = db_path
        self.init_db()
    
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn
    
    def init_db(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Create nodes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        ''')
        
        # Create embeddings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embeddings (
                node_id TEXT PRIMARY KEY,
                vector BLOB,
                dim INTEGER,
                FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE CASCADE
            )
        ''')
        
        # Create edges table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                type TEXT,
                weight REAL,
                created_at TEXT,
                FOREIGN KEY (source) REFERENCES nodes(id) ON DELETE CASCADE,
                FOREIGN KEY (target) REFERENCES nodes(id) ON DELETE CASCADE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_node(self, node_id: str, text: str, metadata: Dict):
        conn = self.get_connection()
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()
        
        cursor.execute('''
            INSERT OR REPLACE INTO nodes (id, text, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (node_id, text, json.dumps(metadata), now, now))
        
        conn.commit()
        conn.close()
    
    def update_node(self, node_id: str, text: Optional[str], metadata: Optional[Dict]):
        conn = self.get_connection()
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()
        
        cursor.execute('SELECT text, metadata FROM nodes WHERE id = ?', (node_id,))
        result = cursor.fetchone()
        if not result:
            conn.close()
            return False
        
        current_text, current_metadata = result
        new_text = text if text is not None else current_text
        new_metadata = metadata if metadata is not None else json.loads(current_metadata)
        
        cursor.execute('''
            UPDATE nodes SET text = ?, metadata = ?, updated_at = ?
            WHERE id = ?
        ''', (new_text, json.dumps(new_metadata), now, node_id))
        
        conn.commit()
        conn.close()
        return True
    
    def get_node(self, node_id: str) -> Optional[Dict]:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, text, metadata, created_at, updated_at FROM nodes WHERE id = ?', (node_id,))
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            return None
        
        node = {
            'id': result[0],
            'text': result[1],
            'metadata': json.loads(result[2]),
            'created_at': result[3],
            'updated_at': result[4]
        }
        
        # Check if has embedding
        cursor.execute('SELECT 1 FROM embeddings WHERE node_id = ?', (node_id,))
        node['has_embedding'] = cursor.fetchone() is not None
        
        # Get edges with weights
        cursor.execute('SELECT id, target, type, weight FROM edges WHERE source = ?', (node_id,))
        outgoing = [
            {'id': r[0], 'target_id': r[1], 'type': r[2], 'weight': r[3]}
            for r in cursor.fetchall()
        ]
        
        cursor.execute('SELECT id, source, type, weight FROM edges WHERE target = ?', (node_id,))
        incoming = [
            {'id': r[0], 'source_id': r[1], 'type': r[2], 'weight': r[3]}
            for r in cursor.fetchall()
        ]
        
        node['edges'] = {'outgoing': outgoing, 'incoming': incoming}
        
        conn.close()
        return node
    
    def delete_node(self, node_id: str) -> bool:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM nodes WHERE id = ?', (node_id,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        return deleted
    
    def get_all_nodes(self) -> List[Dict]:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, text, metadata FROM nodes')
        nodes = []
        for row in cursor.fetchall():
            nodes.append({
                'id': row[0],
                'text': row[1],
                'metadata': json.loads(row[2])
            })
        
        conn.close()
        return nodes
    
    def save_embedding(self, node_id: str, vector: np.ndarray):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        vector_bytes = vector.astype(np.float32).tobytes()
        dim = len(vector)
        
        cursor.execute('''
            INSERT OR REPLACE INTO embeddings (node_id, vector, dim)
            VALUES (?, ?, ?)
        ''', (node_id, vector_bytes, dim))
        
        conn.commit()
        conn.close()
    
    def get_embedding(self, node_id: str) -> Optional[np.ndarray]:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT vector, dim FROM embeddings WHERE node_id = ?', (node_id,))
        result = cursor.fetchone()
        
        conn.close()
        
        if not result:
            return None
        
        vector_bytes, dim = result
        vec = np.frombuffer(vector_bytes, dtype=np.float32)
        return vec
    
    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT node_id, vector, dim FROM embeddings')
        embeddings = {}
        for row in cursor.fetchall():
            node_id, vector_bytes, dim = row
            vec = np.frombuffer(vector_bytes, dtype=np.float32)
            embeddings[node_id] = vec
        conn.close()
        return embeddings
    
    def save_edge(self, edge_id: str, source: str, target: str, edge_type: str, weight: float):
        conn = self.get_connection()
        cursor = conn.cursor()
        now = datetime.utcnow().isoformat()
        
        cursor.execute('''
            INSERT INTO edges (id, source, target, type, weight, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (edge_id, source, target, edge_type, weight, now))
        
        conn.commit()
        conn.close()
    
    def update_edge(self, edge_id: str, edge_type: Optional[str], weight: Optional[float]) -> bool:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT type, weight FROM edges WHERE id = ?', (edge_id,))
        result = cursor.fetchone()
        if not result:
            conn.close()
            return False
        
        current_type, current_weight = result
        new_type = edge_type if edge_type is not None else current_type
        new_weight = weight if weight is not None else current_weight
        
        cursor.execute(
            'UPDATE edges SET type = ?, weight = ? WHERE id = ?',
            (new_type, new_weight, edge_id)
        )
        conn.commit()
        conn.close()
        return True
    
    def delete_edge(self, edge_id: str) -> bool:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM edges WHERE id = ?', (edge_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted
    
    def get_edge(self, edge_id: str) -> Optional[Dict]:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, source, target, type, weight, created_at FROM edges WHERE id = ?', (edge_id,))
        result = cursor.fetchone()
        
        conn.close()
        
        if not result:
            return None
        
        return {
            'id': result[0],
            'source': result[1],
            'target': result[2],
            'type': result[3],
            'weight': result[4],
            'created_at': result[5]
        }
    
    def get_graph_edges(self) -> List[Tuple[str, str, str, float]]:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT source, target, type, weight FROM edges')
        edges = cursor.fetchall()
        
        conn.close()
        return edges


# ============ Embedding Service (hash-based, no models) ============

class EmbeddingService:
    """
    Simple hash-based embedding:
    - Fixed dimension (default 256)
    - Tokenize by lowercase split
    - For each token: index = hash(token) % dim, increment that bin
    - L2-normalize the vector so dot product ~ cosine similarity
    This is fully local, deterministic, and requires no ML models.
    """
    def __init__(self, dim: int = 256):
        self.dim = dim
    
    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()
    
    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = self._tokenize(text)
        for tok in tokens:
            idx = hash(tok) % self.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        return np.vstack([self.encode(t) for t in texts])


# ============ Vector Search Service (no FAISS, simple scan) ============

class VectorSearchService:
    def __init__(self, db_manager: DatabaseManager, embedding_service: EmbeddingService):
        self.db = db_manager
        self.embedding_service = embedding_service
    
    def rebuild_index(self):
        """No-op for now; kept for compatibility with calls."""
        return
    
    def search(self, query_text: str, top_k: int = 10,
               metadata_filter: Optional[Dict[str, str]] = None) -> List[Dict]:
        """Perform vector search by scanning all stored embeddings."""
        embeddings_dict = self.db.get_all_embeddings()
        if not embeddings_dict:
            return []
        
        # Generate query embedding
        query_vec = self.embedding_service.encode(query_text)
        
        # Compute scores
        scores = []
        for node_id, vec in embeddings_dict.items():
            if vec.shape[0] != query_vec.shape[0]:
                continue
            score = float(np.dot(query_vec, vec))
            scores.append((node_id, score))
        
        if not scores:
            return []
        
        # Sort by score desc
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Fetch node details
        nodes = self.db.get_all_nodes()
        node_map = {n['id']: n for n in nodes}
        
        results = []
        for node_id, score in scores:
            node = node_map.get(node_id)
            if not node:
                continue
            # Apply metadata filter if any
            if metadata_filter:
                meta = node.get("metadata", {})
                match = True
                for k, v in metadata_filter.items():
                    if meta.get(k) != v:
                        match = False
                        break
                if not match:
                    continue
            results.append({
                'node': node,
                'vector_score': score
            })
            if len(results) >= top_k:
                break
        
        return results


# ============ BM25 Keyword Search Service (optional) ============

class BM25SearchService:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.bm25 = None
        self.node_ids = []
        self.rebuild_index()
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return text.lower().split()
    
    def rebuild_index(self):
        """Rebuild BM25 index from database"""
        nodes = self.db.get_all_nodes()
        
        if not nodes:
            self.bm25 = None
            self.node_ids = []
            return
        
        self.node_ids = [n['id'] for n in nodes]
        tokenized_texts = [self.tokenize(n['text']) for n in nodes]
        self.bm25 = BM25Okapi(tokenized_texts)
    
    def search(self, query_text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Perform BM25 search, returns list of (node_id, score)"""
        if self.bm25 is None:
            return []
        
        tokenized_query = self.tokenize(query_text)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                results.append((self.node_ids[idx], float(scores[idx])))
        
        return results


# ============ Graph Service ============

class GraphService:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def build_adjacency_list(self, edge_type: Optional[str] = None):
        """
        Build adjacency list from edges, optionally filtered by type.
        Returns: dict[source] = list[(target, weight)]
        """
        edges = self.db.get_graph_edges()
        adj = defaultdict(list)
        
        for source, target, etype, weight in edges:
            if edge_type is None or etype == edge_type:
                adj[source].append((target, weight))
        
        return adj
    
    def bfs_traversal(self, start_id: str, depth: int,
                      edge_type: Optional[str] = None):
        """
        BFS traversal returning:
          distances: {node_id: distance}
          weights:   {node_id: accumulated_weight_along_path}
        """
        adj = self.build_adjacency_list(edge_type=edge_type)
        
        # If start has no edges but exists, just return itself
        if start_id not in adj and not any(
            start_id == t for src, neighs in adj.items() for (t, _) in neighs
        ):
            return {start_id: 0}, {start_id: 0.0}
        
        distances: Dict[str, int] = {start_id: 0}
        weights: Dict[str, float] = {start_id: 0.0}
        queue = deque([(start_id, 0, 0.0)])
        visited = {start_id}
        
        while queue:
            current, dist, acc_w = queue.popleft()
            if dist >= depth:
                continue
            
            for neighbor, w in adj.get(current, []):
                new_dist = dist + 1
                new_w = acc_w + (w if w is not None else 0.0)
                if neighbor not in distances or new_dist < distances[neighbor] or (
                    new_dist == distances[neighbor] and new_w > weights.get(neighbor, -1e9)
                ):
                    distances[neighbor] = new_dist
                    weights[neighbor] = new_w
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, new_dist, new_w))
        
        return distances, weights
    
    def graph_score(
        self,
        start_id: str,
        node_id: str,
        max_depth: int,
        distances: Dict[str, int],
        weights: Dict[str, float]
    ) -> float:
        """
        Calculate graph closeness score, factoring in hop distance and path weight.
        Higher weight paths get higher scores for same depth.
        """
        if node_id == start_id:
            return 1.0
        
        if node_id not in distances:
            return 0.0
        
        d = distances[node_id]
        if d > max_depth:
            return 0.0
        
        # Base depth-based score
        base = (max_depth - d + 1) / (max_depth + 1)
        
        # Normalize weight across all visited nodes
        if weights:
            max_w = max(weights.values())
        else:
            max_w = 0.0
        
        if max_w > 0:
            w_norm = weights.get(node_id, 0.0) / max_w
        else:
            w_norm = 0.0
        
        # Depth score scaled by weight factor in [0.5, 1.0]
        weight_factor = 0.5 + 0.5 * w_norm
        return base * weight_factor
    
    def search_graph(self, start_id: str, depth: int,
                     edge_type: Optional[str] = None) -> Dict:
        """Graph traversal search"""
        distances, weights = self.bfs_traversal(start_id, depth, edge_type=edge_type)
        
        # Get node details
        nodes = []
        for node_id, dist in distances.items():
            if node_id != start_id:  # Exclude start node from results
                node_data = self.db.get_node(node_id)
                if node_data:
                    nodes.append({
                        'node': {
                            'id': node_data['id'],
                            'text': node_data['text'],
                            'metadata': node_data['metadata']
                        },
                        'distance': dist,
                        'path_weight': weights.get(node_id, 0.0)
                    })
        
        # Get edges in the subgraph (include weight and type)
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        node_ids = list(distances.keys())
        placeholders = ','.join('?' * len(node_ids))
        cursor.execute(f'''
            SELECT id, source, target, type, weight FROM edges
            WHERE source IN ({placeholders}) AND target IN ({placeholders})
        ''', node_ids + node_ids)
        
        edges = [
            {
                'id': r[0],
                'source': r[1],
                'target': r[2],
                'type': r[3],
                'weight': r[4]
            }
            for r in cursor.fetchall()
        ]
        
        conn.close()
        
        return {
            'start_id': start_id,
            'depth': depth,
            'edge_type': edge_type,
            'nodes': nodes,
            'edges': edges
        }


# ============ Hybrid Search Service (Vector + Graph) ============

class HybridSearchService:
    def __init__(self, vector_service: VectorSearchService, 
                 bm25_service: BM25SearchService,
                 graph_service: GraphService,
                 db_manager: DatabaseManager):
        self.vector_service = vector_service
        self.bm25_service = bm25_service
        self.graph_service = graph_service
        self.db = db_manager
    
    def hybrid_search(
        self,
        query_text: str,
        vector_weight: float,
        graph_weight: float,
        top_k: int,
        graph_start_id: Optional[str],
        graph_depth: int
    ) -> List[Dict]:
        """
        Perform hybrid search combining:
        - Vector similarity (hash embedding)
        - Graph proximity (depth & weight aware)
        NOTE: BM25 is kept as a separate component but NOT fused into hybrid,
              to ensure vector_weight=1.0 matches /search/vector ordering.
        """

        print(
            "[HYBRID DEBUG] v_w=", vector_weight,
            "g_w=", graph_weight,
            "top_k=", top_k,
            "graph_start_id=", graph_start_id,
            "graph_depth=", graph_depth,
        )

        all_nodes = self.db.get_all_nodes()
        if not all_nodes:
            return []
        
        node_map = {n['id']: n for n in all_nodes}
        
        # 1. Vector search rankings (no metadata filter here)
        vector_results = self.vector_service.search(
            query_text, top_k=len(all_nodes), metadata_filter=None
        )
        # Build raw vector scores
        vec_scores = {r['node']['id']: r['vector_score'] for r in vector_results}
        
        # Normalize vector scores to [0, 1]
        if vec_scores:
            max_vec = max(vec_scores.values())
            min_vec = min(vec_scores.values())
            if max_vec > min_vec:
                vec_scores = {
                    k: (v - min_vec) / (max_vec - min_vec)
                    for k, v in vec_scores.items()
                }
            else:
                vec_scores = {k: 1.0 for k in vec_scores.keys()}
        
        # 2. Graph proximity scores
        graph_scores: Dict[str, float] = {nid: 0.0 for nid in node_map.keys()}
        if graph_start_id:
            distances, weights = self.graph_service.bfs_traversal(
                graph_start_id, graph_depth, edge_type=None
            )
            for node_id in node_map.keys():
                graph_scores[node_id] = self.graph_service.graph_score(
                    graph_start_id, node_id, graph_depth, distances, weights
                )
        
        # 3. Combine with weights
        final_results = []
        for node_id, node in node_map.items():
            text_score = vec_scores.get(node_id, 0.0)
            g_score = graph_scores.get(node_id, 0.0)
            
            final_score = vector_weight * text_score + graph_weight * g_score
            
            if final_score > 0:
                final_results.append({
                    'node': node,
                    'vector_score': text_score,
                    'graph_score': g_score,
                    'final_score': final_score
                })
        
        final_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        return final_results[:top_k]

# ============ FastAPI Application ============

app = FastAPI(title="Vector + Graph Native Database")

# Initialize services
db_manager = DatabaseManager()
embedding_service = EmbeddingService()
vector_service = VectorSearchService(db_manager, embedding_service)
bm25_service = BM25SearchService(db_manager)
graph_service = GraphService(db_manager)
hybrid_service = HybridSearchService(vector_service, bm25_service, graph_service, db_manager)


# ============ Node Endpoints ============

@app.post("/nodes", status_code=201)
def create_node(node: NodeCreate):
    """Create a new node"""
    node_id = f"node-{datetime.utcnow().timestamp()}"
    
    # Save node text + metadata
    db_manager.save_node(node_id, node.text, node.metadata)
    
    # Decide embedding:
    emb_vec: Optional[np.ndarray] = None
    if node.embedding is not None:
        emb_vec = np.array(node.embedding, dtype=np.float32)
    elif node.auto_embed:
        emb_vec = embedding_service.encode(node.text)
    
    if emb_vec is not None:
        db_manager.save_embedding(node_id, emb_vec)
    
    # Rebuild indices
    vector_service.rebuild_index()
    bm25_service.rebuild_index()
    
    node_data = db_manager.get_node(node_id)
    # Include embedding in response as required by tests
    if emb_vec is None:
        emb_vec = db_manager.get_embedding(node_id)
    if emb_vec is not None:
        node_data["embedding"] = emb_vec.tolist()
    
    return node_data


@app.get("/nodes/{node_id}")
def get_node(node_id: str):
    """Get a node by ID"""
    node = db_manager.get_node(node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return node


@app.put("/nodes/{node_id}")
def update_node(node_id: str, update: NodeUpdate):
    """Update a node"""
    success = db_manager.update_node(node_id, update.text, update.metadata)
    
    if not success:
        raise HTTPException(status_code=404, detail="Node not found")
    
    # Handle embedding update:
    emb_vec: Optional[np.ndarray] = None
    if update.embedding is not None:
        emb_vec = np.array(update.embedding, dtype=np.float32)
    elif update.regen_embedding:
        node_data = db_manager.get_node(node_id)
        emb_vec = embedding_service.encode(node_data['text'])
    
    if emb_vec is not None:
        db_manager.save_embedding(node_id, emb_vec)
        vector_service.rebuild_index()
        bm25_service.rebuild_index()
    
    node_data = db_manager.get_node(node_id)
    if emb_vec is None:
        emb_vec = db_manager.get_embedding(node_id)
    if emb_vec is not None:
        node_data["embedding"] = emb_vec.tolist()
    
    return node_data


@app.delete("/nodes/{node_id}", status_code=204)
def delete_node(node_id: str):
    """Delete a node (and cascading edges via FK)"""
    success = db_manager.delete_node(node_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Node not found")
    
    vector_service.rebuild_index()
    bm25_service.rebuild_index()
    return None


# ============ Edge Endpoints ============

@app.post("/edges", status_code=201)
def create_edge(edge: EdgeCreate):
    """Create a new edge"""
    # Verify nodes exist
    if not db_manager.get_node(edge.source):
        raise HTTPException(status_code=404, detail=f"Source node {edge.source} not found")
    if not db_manager.get_node(edge.target):
        raise HTTPException(status_code=404, detail=f"Target node {edge.target} not found")
    
    edge_id = f"edge-{datetime.utcnow().timestamp()}"
    db_manager.save_edge(edge_id, edge.source, edge.target, edge.type, edge.weight)
    return db_manager.get_edge(edge_id)


@app.get("/edges/{edge_id}")
def get_edge(edge_id: str):
    """Get an edge by ID"""
    edge = db_manager.get_edge(edge_id)
    if not edge:
        raise HTTPException(status_code=404, detail="Edge not found")
    return edge


@app.put("/edges/{edge_id}")
def update_edge(edge_id: str, update: EdgeUpdate):
    """Update edge type/weight"""
    success = db_manager.update_edge(edge_id, update.type, update.weight)
    if not success:
        raise HTTPException(status_code=404, detail="Edge not found")
    return db_manager.get_edge(edge_id)


@app.delete("/edges/{edge_id}", status_code=204)
def delete_edge(edge_id: str):
    """Delete an edge"""
    success = db_manager.delete_edge(edge_id)
    if not success:
        raise HTTPException(status_code=404, detail="Edge not found")
    return None


# ============ Search Endpoints ============

@app.post("/search/vector")
def search_vector(request: VectorSearchRequest):
    """Perform vector similarity search (optionally filtered by metadata)"""
    results = vector_service.search(
        request.query_text,
        request.top_k,
        metadata_filter=request.metadata_filter
    )
    return results


@app.get("/search/graph")
def search_graph(start_id: str, depth: int, type: Optional[str] = None):
    """
    Perform graph traversal search.
    'type' parameter filters edges by relationship type when provided.
    """
    if not db_manager.get_node(start_id):
        raise HTTPException(status_code=404, detail="Start node not found")
    
    results = graph_service.search_graph(start_id, depth, edge_type=type)
    return results


@app.post("/search/hybrid")
def search_hybrid(request: HybridSearchRequest):
    """Perform hybrid search combining vector similarity and graph proximity"""
    if request.graph_start_id and not db_manager.get_node(request.graph_start_id):
        raise HTTPException(status_code=404, detail="Graph start node not found")
    
    results = hybrid_service.hybrid_search(
        request.query_text,
        request.vector_weight,
        request.graph_weight,
        request.top_k,
        request.graph_start_id,
        request.graph_depth
    )
    return results


@app.get("/")
def root():
    """Health check"""
    return {"status": "Vector + Graph Database is running"}


@app.get("/stats")
def get_stats():
    """Get database statistics"""
    nodes = db_manager.get_all_nodes()
    embeddings = db_manager.get_all_embeddings()
    edges = db_manager.get_graph_edges()
    
    return {
        "total_nodes": len(nodes),
        "nodes_with_embeddings": len(embeddings),
        "total_edges": len(edges)
    }


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("APP_HOST", "127.0.0.1")
    port = int(os.getenv("APP_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
