import requests
import uuid
import math

BASE_URL = "http://127.0.0.1:8000"


# ============ Helpers ============

def approx_equal(a, b, eps=1e-6):
    return abs(a - b) <= eps


def cosine_similarity(v1, v2):
    import numpy as np
    a = np.array(v1, dtype=float)
    b = np.array(v2, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def create_node(text, metadata=None, auto_embed=True, embedding=None):
    if metadata is None:
        metadata = {}
    payload = {
        "text": text,
        "metadata": metadata,
        "auto_embed": auto_embed,
    }
    if embedding is not None:
        payload["embedding"] = embedding
    resp = requests.post(f"{BASE_URL}/nodes", json=payload)
    resp.raise_for_status()
    return resp.json()


def get_node(node_id):
    return requests.get(f"{BASE_URL}/nodes/{node_id}")


def update_node(node_id, text=None, metadata=None, regen_embedding=False, embedding=None):
    payload = {
        "regen_embedding": regen_embedding,
    }
    if text is not None:
        payload["text"] = text
    if metadata is not None:
        payload["metadata"] = metadata
    if embedding is not None:
        payload["embedding"] = embedding
    resp = requests.put(f"{BASE_URL}/nodes/{node_id}", json=payload)
    return resp


def delete_node(node_id):
    return requests.delete(f"{BASE_URL}/nodes/{node_id}")


def create_edge(source_id, target_id, edge_type="default", weight=1.0):
    payload = {
        "source": source_id,
        "target": target_id,
        "type": edge_type,
        "weight": weight,
    }
    resp = requests.post(f"{BASE_URL}/edges", json=payload)
    resp.raise_for_status()
    return resp.json()


def get_edge(edge_id):
    return requests.get(f"{BASE_URL}/edges/{edge_id}")


def update_edge(edge_id, edge_type=None, weight=None):
    payload = {}
    if edge_type is not None:
        payload["type"] = edge_type
    if weight is not None:
        payload["weight"] = weight
    resp = requests.put(f"{BASE_URL}/edges/{edge_id}", json=payload)
    return resp


def delete_edge(edge_id):
    return requests.delete(f"{BASE_URL}/edges/{edge_id}")


def search_vector(query_text, top_k=10, metadata_filter=None):
    payload = {"query_text": query_text, "top_k": top_k}
    if metadata_filter is not None:
        payload["metadata_filter"] = metadata_filter
    resp = requests.post(f"{BASE_URL}/search/vector", json=payload)
    resp.raise_for_status()
    return resp.json()


def search_graph(start_id, depth, edge_type=None):
    params = {"start_id": start_id, "depth": depth}
    if edge_type is not None:
        params["type"] = edge_type
    resp = requests.get(f"{BASE_URL}/search/graph", params=params)
    resp.raise_for_status()
    return resp.json()


def search_hybrid(query_text, vector_weight, graph_weight, top_k,
                  graph_start_id=None, graph_depth=2):
    payload = {
        "query_text": query_text,
        "vector_weight": vector_weight,
        "graph_weight": graph_weight,
        "top_k": top_k,
        "graph_start_id": graph_start_id,
        "graph_depth": graph_depth,
    }
    resp = requests.post(f"{BASE_URL}/search/hybrid", json=payload)
    resp.raise_for_status()
    return resp.json()


def get_stats():
    resp = requests.get(f"{BASE_URL}/stats")
    resp.raise_for_status()
    return resp.json()


def node_id_from_result(item):
    node = item.get("node", {})
    return node.get("id")


def node_metadata_from_result(item):
    node = item.get("node", {})
    return node.get("metadata", {}) or {}


# ============ 1. API & CRUD Tests ============

def tc_api_01():
    """
    TC-API-01 (P0) — Create node
    POST /nodes with text + metadata, expect 201 + id + embedding; GET returns same text/metadata.
    """
    print("\n[TC-API-01] Create node")
    payload_text = "Venkat's note on caching"
    payload_meta = {"type": "note", "author": "v", "tc": "TC-API-01"}

    node = create_node(payload_text, payload_meta)
    node_id = node["id"]

    ok = True
    if node["text"] != payload_text:
        print("  FAIL: POST returned different text.")
        ok = False
    meta = node.get("metadata", {})
    for k, v in payload_meta.items():
        if meta.get(k) != v:
            print(f"  FAIL: Metadata mismatch for key {k}: {meta.get(k)} != {v}")
            ok = False
    if "embedding" not in node:
        print("  FAIL: embedding field missing in POST /nodes response.")
        ok = False

    # GET should return same text and metadata (embedding may or may not be included)
    resp_get = get_node(node_id)
    if resp_get.status_code != 200:
        print("  FAIL: GET /nodes/{id} did not return 200.")
        ok = False
    else:
        data = resp_get.json()
        if data["text"] != payload_text:
            print("  FAIL: GET returned different text.")
            ok = False
        meta2 = data.get("metadata", {})
        for k, v in payload_meta.items():
            if meta2.get(k) != v:
                print(f"  FAIL: GET metadata mismatch for key {k}.")
                ok = False

    if ok:
        print("  PASS")
    return ok


def tc_api_02():
    """
    TC-API-02 (P0) — Read node with relationships
    Create A and B; create edge A->B; GET A and expect outgoing edge listed.
    """
    print("\n[TC-API-02] Read node with relationships (edges)")
    A = create_node("Node A for TC-API-02", {"tc": "TC-API-02", "label": "A"})
    B = create_node("Node B for TC-API-02", {"tc": "TC-API-02", "label": "B"})
    A_id = A["id"]
    B_id = B["id"]

    edge = create_edge(A_id, B_id, edge_type="api-02-edge", weight=2.5)

    resp = get_node(A_id)
    ok = True
    if resp.status_code != 200:
        print("  FAIL: GET /nodes/A returned non-200.")
        return False
    data = resp.json()
    edges = data.get("edges", {})
    outgoing = edges.get("outgoing", [])
    found = False
    for e in outgoing:
        if e.get("target_id") == B_id and e.get("type") == "api-02-edge":
            found = True
            if not approx_equal(e.get("weight", 0.0), 2.5):
                print("  FAIL: Edge weight mismatch.")
                ok = False
            break
    if not found:
        print("  FAIL: Outgoing edge A->B not listed in node A.")
        ok = False

    if ok:
        print("  PASS")
    return ok


def tc_api_03():
    """
    TC-API-03 (P0) — Update node & re-generate embedding
    - Create node with text1
    - Capture embedding
    - PUT with new text2 and regen_embedding=true
    - Expect embedding changed (cosine < 0.99) and text updated.
    """
    print("\n[TC-API-03] Update node & re-generate embedding")
    text1 = "Deep learning for image classification"
    text2 = "Classical music orchestras and symphonies"
    node = create_node(text1, {"tc": "TC-API-03"})
    node_id = node["id"]
    old_emb = node.get("embedding")
    if old_emb is None:
        print("  FAIL: No embedding in create_node for TC-API-03.")
        return False

    resp_put = update_node(node_id, text=text2, regen_embedding=True)
    ok = True
    if resp_put.status_code != 200:
        print("  FAIL: PUT /nodes/{id} returned non-200.")
        return False
    updated = resp_put.json()

    if updated["text"] != text2:
        print("  FAIL: Updated text incorrect.")
        ok = False

    new_emb = updated.get("embedding")
    if new_emb is None:
        print("  FAIL: Updated node missing embedding in response.")
        ok = False
    else:
        cos = cosine_similarity(old_emb, new_emb)
        if cos >= 0.99:
            print(f"  FAIL: Embedding not sufficiently changed (cos={cos}).")
            ok = False

    # Double-check via GET
    resp_get = get_node(node_id)
    if resp_get.status_code != 200 or resp_get.json().get("text") != text2:
        print("  FAIL: GET after update did not return new text.")
        ok = False

    if ok:
        print("  PASS")
    return ok


def tc_api_04():
    """
    TC-API-04 (P0) — Delete node cascading edges
    - Create two nodes X, Y
    - Create edges X->Y and Y->X
    - DELETE X
    - Verify GET X is 404 and edges referencing X are gone (404).
    """
    print("\n[TC-API-04] Delete node cascading edges")
    X = create_node("Node X for TC-API-04", {"tc": "TC-API-04", "label": "X"})
    Y = create_node("Node Y for TC-API-04", {"tc": "TC-API-04", "label": "Y"})
    X_id = X["id"]
    Y_id = Y["id"]

    e1 = create_edge(X_id, Y_id, edge_type="api-04", weight=1.0)
    e2 = create_edge(Y_id, X_id, edge_type="api-04", weight=2.0)
    e1_id = e1["id"]
    e2_id = e2["id"]

    resp_del = delete_node(X_id)
    ok = True
    if resp_del.status_code != 204:
        print("  FAIL: DELETE /nodes/X did not return 204.")
        ok = False

    if get_node(X_id).status_code != 404:
        print("  FAIL: GET /nodes/X after delete did not return 404.")
        ok = False

    if get_edge(e1_id).status_code != 404:
        print("  FAIL: Edge e1 still exists after deleting node X.")
        ok = False
    if get_edge(e2_id).status_code != 404:
        print("  FAIL: Edge e2 still exists after deleting node X.")
        ok = False

    if ok:
        print("  PASS")
    return ok


def tc_api_05():
    """
    TC-API-05 (P1) — Relationship CRUD
    - Create A, B
    - POST /edges
    - GET /edges/{id}
    - PUT to update weight
    - GET verifies new weight
    - Also verify traversal reflects new weight via /search/graph.
    """
    print("\n[TC-API-05] Relationship CRUD")
    A = create_node("Node A for TC-API-05", {"tc": "TC-API-05", "label": "A"})
    B = create_node("Node B for TC-API-05", {"tc": "TC-API-05", "label": "B"})
    A_id = A["id"]
    B_id = B["id"]

    edge = create_edge(A_id, B_id, edge_type="api-05", weight=1.0)
    edge_id = edge["id"]

    resp_get_e = get_edge(edge_id)
    ok = True
    if resp_get_e.status_code != 200:
        print("  FAIL: GET /edges/{id} returned non-200.")
        ok = False
    else:
        data = resp_get_e.json()
        if data["source"] != A_id or data["target"] != B_id:
            print("  FAIL: Edge endpoints differ from expected.")
            ok = False
        if data["type"] != "api-05":
            print("  FAIL: Edge type mismatch.")
            ok = False
        if not approx_equal(data["weight"], 1.0):
            print("  FAIL: Edge weight mismatch.")
            ok = False

    # Update weight
    resp_put = update_edge(edge_id, weight=3.5)
    if resp_put.status_code != 200:
        print("  FAIL: PUT /edges/{id} returned non-200.")
        ok = False
    else:
        updated_edge = get_edge(edge_id).json()
        if not approx_equal(updated_edge["weight"], 3.5):
            print("  FAIL: Edge weight not updated.")
            ok = False

    # Check traversal
    graph_res = search_graph(A_id, depth=1, edge_type="api-05")
    nodes = graph_res.get("nodes", [])
    found_B = False
    for n in nodes:
        nid = n["node"]["id"]
        if nid == B_id:
            found_B = True
            if not approx_equal(n.get("path_weight", 0.0), 3.5):
                print("  FAIL: Graph traversal path_weight did not reflect updated edge weight.")
                ok = False
            break
    if not found_B:
        print("  FAIL: Graph traversal did not include node B.")
        ok = False

    if ok:
        print("  PASS")
    return ok


# ============ 3. Vector Search Tests ============

def tc_vec_01():
    """
    TC-VEC-01 (P0) — Top-k cosine similarity ordering
    Insert three nodes A (very similar), B (medium), C (far).
    Expect order A, B, C among results.
    """
    print("\n[TC-VEC-01] Top-k cosine similarity ordering")
    query = "deep learning image classification convolutional neural networks"

    A = create_node(
        "Convolutional neural networks for deep learning image classification tasks",
        {"tc": "TC-VEC-01", "label": "A"}
    )
    B = create_node(
        "Deep learning methods for audio classification and speech processing",
        {"tc": "TC-VEC-01", "label": "B"}
    )
    C = create_node(
        "Classical music orchestras and symphonies",
        {"tc": "TC-VEC-01", "label": "C"}
    )

    A_id = A["id"]
    B_id = B["id"]
    C_id = C["id"]

    results = search_vector(query, top_k=20)
    ids = [node_id_from_result(r) for r in results]

    ok = True
    try:
        idxA = ids.index(A_id)
        idxB = ids.index(B_id)
        idxC = ids.index(C_id)
    except ValueError:
        print("  FAIL: Not all A,B,C found in vector search results.")
        return False

    if not (idxA < idxB < idxC):
        print(f"  FAIL: Order is not A<B<C; got indices A={idxA}, B={idxB}, C={idxC}.")
        ok = False

    # check top result similarity threshold (rough)
    top_for_A = None
    for r in results:
        if node_id_from_result(r) == A_id:
            top_for_A = r["vector_score"]
            break
    if top_for_A is not None and top_for_A < 0.5:
        print(f"  FAIL: Top similarity for A is too low: {top_for_A}")
        ok = False

    if ok:
        print("  PASS")
    return ok


def tc_vec_02():
    """
    TC-VEC-02 (P1) — Top-k with k > dataset size
    Expect: returns all items with embeddings, count = nodes_with_embeddings.
    """
    print("\n[TC-VEC-02] Top-k with k > dataset size")
    stats = get_stats()
    n_emb = stats.get("nodes_with_embeddings", 0)

    results = search_vector("random query for TC-VEC-02", top_k=n_emb + 10)
    count = len(results)

    ok = True
    if count != n_emb:
        print(f"  FAIL: Expected {n_emb} results, got {count}.")
        ok = False

    if ok:
        print("  PASS")
    return ok


def tc_vec_03():
    """
    TC-VEC-03 (P1) — Filtering by metadata
    Create notes and non-notes; filter metadata.type=note; expect only notes in results.
    """
    print("\n[TC-VEC-03] Filtering by metadata")
    create_node("Caching note one", {"tc": "TC-VEC-03", "type": "note"})
    create_node("Caching note two", {"tc": "TC-VEC-03", "type": "note"})
    create_node("Some random paper", {"tc": "TC-VEC-03", "type": "paper"})

    results = search_vector(
        "caching note for TC-VEC-03",
        top_k=10,
        metadata_filter={"type": "note"}
    )

    ok = True
    if not results:
        print("  FAIL: No results returned with metadata filter type=note.")
        return False

    for r in results:
        meta = node_metadata_from_result(r)
        if meta.get("type") != "note":
            print("  FAIL: Result with metadata.type != 'note' returned.")
            ok = False

    if ok:
        print("  PASS")
    return ok


# ============ 4. Graph Traversal Tests ============

def tc_graph_01():
    """
    TC-GRAPH-01 (P0) — BFS / depth-limited traversal
    Build chain A->B->C->D; depth=2 from A should return B and C, not D.
    """
    print("\n[TC-GRAPH-01] BFS / depth-limited traversal")
    A = create_node("Graph A", {"tc": "TC-GRAPH-01", "label": "A"})
    B = create_node("Graph B", {"tc": "TC-GRAPH-01", "label": "B"})
    C = create_node("Graph C", {"tc": "TC-GRAPH-01", "label": "C"})
    D = create_node("Graph D", {"tc": "TC-GRAPH-01", "label": "D"})
    A_id, B_id, C_id, D_id = A["id"], B["id"], C["id"], D["id"]

    create_edge(A_id, B_id, edge_type="graph-01")
    create_edge(B_id, C_id, edge_type="graph-01")
    create_edge(C_id, D_id, edge_type="graph-01")

    res = search_graph(A_id, depth=2, edge_type="graph-01")
    nodes = res.get("nodes", [])
    found_ids = {n["node"]["id"] for n in nodes}

    ok = True
    if B_id not in found_ids or C_id not in found_ids:
        print("  FAIL: Did not find B and C in depth-limited traversal.")
        ok = False
    if D_id in found_ids:
        print("  FAIL: D should not be included at depth=2 from A.")
        ok = False

    if ok:
        print("  PASS")
    return ok


def tc_graph_02():
    """
    TC-GRAPH-02 (P1) — Multi-type relationships
    When filtered by type=author_of, only those edges are followed.
    """
    print("\n[TC-GRAPH-02] Multi-type relationships")
    A = create_node("Author node", {"tc": "TC-GRAPH-02", "label": "A"})
    P1 = create_node("Paper 1", {"tc": "TC-GRAPH-02", "label": "P1"})
    P2 = create_node("Paper 2", {"tc": "TC-GRAPH-02", "label": "P2"})
    A_id, P1_id, P2_id = A["id"], P1["id"], P2["id"]

    create_edge(A_id, P1_id, edge_type="author_of", weight=1.0)
    create_edge(A_id, P2_id, edge_type="cites", weight=1.0)

    res = search_graph(A_id, depth=1, edge_type="author_of")
    nodes = res.get("nodes", [])
    found_ids = {n["node"]["id"] for n in nodes}

    ok = True
    if P1_id not in found_ids:
        print("  FAIL: P1 not found under type=author_of.")
        ok = False
    if P2_id in found_ids:
        print("  FAIL: P2 should not appear when filtering type=author_of.")
        ok = False

    if ok:
        print("  PASS")
    return ok


def tc_graph_03():
    """
    TC-GRAPH-03 (P1) — Cycle handling
    Graph has cycles A->B->A; traversal must not infinite-loop.
    Expect nodes visited once; traversal terminates; only A,B returned.
    """
    print("\n[TC-GRAPH-03] Cycle handling")
    A = create_node("Cycle A", {"tc": "TC-GRAPH-03", "label": "A"})
    B = create_node("Cycle B", {"tc": "TC-GRAPH-03", "label": "B"})
    A_id, B_id = A["id"], B["id"]

    create_edge(A_id, B_id, edge_type="cycle", weight=1.0)
    create_edge(B_id, A_id, edge_type="cycle", weight=1.0)

    res = search_graph(A_id, depth=3, edge_type="cycle")
    nodes = res.get("nodes", [])
    found_ids = {n["node"]["id"] for n in nodes}

    ok = True
    if not (found_ids <= {A_id, B_id}):
        print(f"  FAIL: Unexpected nodes in cycle traversal: {found_ids}")
        ok = False
    if A_id in found_ids:
        # start node should be excluded per implementation
        print("  FAIL: Start node A should not appear in nodes list.")
        ok = False

    if ok:
        print("  PASS")
    return ok


# ============ 5. Hybrid Search Tests ============

def setup_hybrid_test_nodes():
    """
    Create three fresh nodes for hybrid tests:
    - V-similar: high vector score to the query, low graph score (no edge)
    - G-close: low vector score, graph-close to Neutral
    - Neutral: graph_start node with an edge only to G-close
    """
    suffix = str(uuid.uuid4())[:8]
    query_text = "deep learning image classification convolutional neural networks"

    v_text = (
        "Convolutional neural networks are widely used in deep learning "
        "for image classification tasks."
    )

    g_text = "Classical music orchestras and symphonies performing baroque pieces."

    neutral_text = "Neutral root node for hybrid search tests."

    V = create_node(v_text, {"role": "V-similar", "test_suite": "hyb-tests", "suffix": suffix})
    G = create_node(g_text, {"role": "G-close", "test_suite": "hyb-tests", "suffix": suffix})
    N = create_node(neutral_text, {"role": "Neutral", "test_suite": "hyb-tests", "suffix": suffix})

    create_edge(N["id"], G["id"], edge_type="hyb-tests-edge", weight=1.0)

    return {
        "query_text": query_text,
        "v_id": V["id"],
        "g_id": G["id"],
        "neutral_id": N["id"],
    }


def tc_hyb_01(context=None):
    """
    TC-HYB-01 (P0) — Weighted merge correctness
    vector_weight=0.7, graph_weight=0.3
    Expected: V-similar ranks above G-close; final_score matches weighted formula.
    """
    print("\n[TC-HYB-01] Weighted merge correctness")
    if context is None:
        context = setup_hybrid_test_nodes()

    query_text = context["query_text"]
    v_id = context["v_id"]
    g_id = context["g_id"]
    neutral_id = context["neutral_id"]

    results = search_hybrid(
        query_text=query_text,
        vector_weight=0.7,
        graph_weight=0.3,
        top_k=50,
        graph_start_id=neutral_id,
        graph_depth=2,
    )

    v_idx = g_idx = None
    v_item = g_item = None

    for idx, item in enumerate(results):
        nid = node_id_from_result(item)
        if nid == v_id:
            v_idx = idx
            v_item = item
        elif nid == g_id:
            g_idx = idx
            g_item = item

    ok = True
    if v_idx is None or g_idx is None:
        print("  FAIL: Could not find both V-similar and G-close in hybrid results.")
        return False

    ordering_ok = v_idx < g_idx

    vf = v_item.get("final_score")
    vv = v_item.get("vector_score", 0.0)
    vg = v_item.get("graph_score", 0.0)
    gf = g_item.get("final_score")
    gv = g_item.get("vector_score", 0.0)
    gg = g_item.get("graph_score", 0.0)

    v_formula_ok = approx_equal(vf, 0.7 * vv + 0.3 * vg)
    g_formula_ok = approx_equal(gf, 0.7 * gv + 0.3 * gg)

    print(f"  V-similar index={v_idx}, final={vf}, vec={vv}, graph={vg}")
    print(f"  G-close   index={g_idx}, final={gf}, vec={gv}, graph={gg}")

    if not ordering_ok:
        print("  FAIL: V-similar did not rank above G-close.")
        ok = False
    if not (v_formula_ok and g_formula_ok):
        print("  FAIL: final_score does not match weighted formula.")
        ok = False

    if ok:
        print("  PASS")
    return ok


def tc_hyb_02(context=None):
    """
    TC-HYB-02 (P0) — Tuning extremes
    - vector_weight=1.0, graph_weight=0.0 => matches vector-only ordering
    - vector_weight=0.0, graph_weight=1.0 => graph-only proximity ordering
    """
    print("\n[TC-HYB-02] Tuning extremes")
    if context is None:
        context = setup_hybrid_test_nodes()

    query_text = context["query_text"]
    v_id = context["v_id"]
    g_id = context["g_id"]
    neutral_id = context["neutral_id"]

    # Part A: vector-only
    vec_results = search_vector(query_text, top_k=50)
    vec_order = [node_id_from_result(r) for r in vec_results]

    hybr_vec = search_hybrid(
        query_text=query_text,
        vector_weight=1.0,
        graph_weight=0.0,
        top_k=50,
        graph_start_id=neutral_id,
        graph_depth=2,
    )
    hybr_vec_order = [node_id_from_result(r) for r in hybr_vec]

    n = min(len(vec_order), len(hybr_vec_order))
    vec_match = (vec_order[:n] == hybr_vec_order[:n])

    if vec_match:
        print("  PASS (A): vector_weight=1.0, graph_weight=0.0 matches vector-only ordering.")
    else:
        print("  FAIL (A): vector-only and hybrid(vector_weight=1.0) orderings differ.")
        print("    Vector order (first few):", vec_order[:5])
        print("    Hybrid  order (first few):", hybr_vec_order[:5])

    # Part B: graph-only
    hybr_graph = search_hybrid(
        query_text=query_text,
        vector_weight=0.0,
        graph_weight=1.0,
        top_k=100,
        graph_start_id=neutral_id,
        graph_depth=2,
    )

    g_idx = v_idx = None
    g_item = v_item = None
    for idx, item in enumerate(hybr_graph):
        nid = node_id_from_result(item)
        if nid == g_id:
            g_idx = idx
            g_item = item
        elif nid == v_id:
            v_idx = idx
            v_item = item

    graph_ok = True
    if g_idx is None:
        print("  FAIL (B): G-close not returned in graph-only hybrid results.")
        graph_ok = False
    else:
        # final_score should equal graph_score for each item
        all_formula_ok = True
        for item in hybr_graph:
            gf = item.get("final_score")
            gg = item.get("graph_score", 0.0)
            if not approx_equal(gf, gg):
                print("  FAIL (B): final_score != graph_score for some item.")
                all_formula_ok = False
                break

        if v_idx is not None:
            score_relation_ok = g_idx < v_idx
        else:
            # If V-similar has graph_score=0 -> final_score=0, may be at bottom; we require G-close near top.
            score_relation_ok = (g_idx == 0)

        if g_item is not None:
            print(f"  G-close index={g_idx}, final={g_item.get('final_score')}, graph={g_item.get('graph_score')}")
        if v_item is not None:
            print(f"  V-similar index={v_idx}, final={v_item.get('final_score')}, graph={v_item.get('graph_score')}")

        if not score_relation_ok:
            print("  FAIL (B): G-close does not outrank V-similar in graph-only mode.")
        if not all_formula_ok:
            graph_ok = False
        else:
            graph_ok = graph_ok and score_relation_ok

    all_ok = vec_match and graph_ok
    if all_ok:
        print("  PASS (overall)")
    return all_ok


# ============ Main Runner ============

def main():
    # Quick connectivity check
    try:
        health = requests.get(f"{BASE_URL}/").json()
        print("Server health:", health)
    except Exception as e:
        print("Cannot reach API at", BASE_URL)
        print("Make sure uvicorn is running, e.g.:")
        print("  uvicorn app:app --reload --host 127.0.0.1 --port 8000")
        return

    results = {}

    # 1. API & CRUD
    results["TC-API-01"] = tc_api_01()
    results["TC-API-02"] = tc_api_02()
    results["TC-API-03"] = tc_api_03()
    results["TC-API-04"] = tc_api_04()
    results["TC-API-05"] = tc_api_05()

    # 3. Vector Search
    results["TC-VEC-01"] = tc_vec_01()
    results["TC-VEC-02"] = tc_vec_02()
    results["TC-VEC-03"] = tc_vec_03()

    # 4. Graph Traversal
    results["TC-GRAPH-01"] = tc_graph_01()
    results["TC-GRAPH-02"] = tc_graph_02()
    results["TC-GRAPH-03"] = tc_graph_03()

    # 5. Hybrid Search
    hyb_context = setup_hybrid_test_nodes()
    results["TC-HYB-01"] = tc_hyb_01(hyb_context)
    results["TC-HYB-02"] = tc_hyb_02(hyb_context)

    print("\n================ Summary ================")
    for name, ok in results.items():
        print(f"{name}: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
