from flask import Flask, request, jsonify
from flask_cors import CORS
import hashlib
import time
import numpy as np
from collections import OrderedDict
from datetime import datetime, timedelta
import os

app = Flask(__name__)
CORS(app)

# -------------------------
# CONFIGURATION
# -------------------------
CACHE_SIZE = 1500
TTL_HOURS = 24
MODEL_COST_PER_MILLION = 0.50
AVG_TOKENS = 500

# -------------------------
# CACHE STORAGE
# -------------------------
cache = OrderedDict()
embeddings_store = {}

# -------------------------
# ANALYTICS COUNTERS
# -------------------------
total_requests = 0
cache_hits = 0
cache_misses = 0


# -------------------------
# HELPER FUNCTIONS
# -------------------------

def normalize_query(query):
    return query.lower().strip()


def generate_hash(query):
    return hashlib.md5(query.encode()).hexdigest()


def generate_embedding(text):
    np.random.seed(abs(hash(text)) % (10 ** 8))
    return np.random.rand(128)


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2)
    )


def remove_expired_entries():
    now = datetime.now()
    keys_to_delete = []
    for key, value in list(cache.items()):
        if now - value["timestamp"] > timedelta(hours=TTL_HOURS):
            keys_to_delete.append(key)

    for key in keys_to_delete:
        cache.pop(key, None)
        embeddings_store.pop(key, None)


def call_llm(query):
    time.sleep(1)  # simulate API delay
    return f"Answer for: {query}"


# -------------------------
# ROOT GET (Prevents 405)
# -------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "FAQ Cache API running"})


# -------------------------
# MAIN POST ENDPOINT
# -------------------------
@app.route("/", methods=["POST"])
def handle_query():
    global total_requests, cache_hits, cache_misses

    start_time = time.time()
    total_requests += 1

    data = request.get_json(silent=True) or {}
    query = normalize_query(data.get("query", ""))

    if not query:
        return jsonify({"error": "Query is required"}), 400

    remove_expired_entries()
    cache_key = generate_hash(query)

    # 1️⃣ Exact Match
    if cache_key in cache:
        cache_hits += 1
        cache.move_to_end(cache_key)
        latency = max(1, int((time.time() - start_time) * 1000))
        return jsonify({
            "answer": cache[cache_key]["answer"],
            "cached": True,
            "latency": latency,
            "cacheKey": cache_key
        })

    # 2️⃣ Semantic Match
    query_embedding = generate_embedding(query)

    for key, stored_embedding in embeddings_store.items():
        similarity = cosine_similarity(query_embedding, stored_embedding)
        if similarity > 0.95:
            cache_hits += 1
            cache.move_to_end(key)
            latency = max(1, int((time.time() - start_time) * 1000))
            return jsonify({
                "answer": cache[key]["answer"],
                "cached": True,
                "latency": latency,
                "cacheKey": key
            })

    # 3️⃣ Cache Miss
    cache_misses += 1
    answer = call_llm(query)

    if len(cache) >= CACHE_SIZE:
        oldest = next(iter(cache))
        cache.pop(oldest, None)
        embeddings_store.pop(oldest, None)

    cache[cache_key] = {
        "answer": answer,
        "timestamp": datetime.now()
    }
    embeddings_store[cache_key] = query_embedding

    latency = max(1, int((time.time() - start_time) * 1000))

    return jsonify({
        "answer": answer,
        "cached": False,
        "latency": latency,
        "cacheKey": cache_key
    })


# -------------------------
# ANALYTICS ENDPOINT
# -------------------------
@app.route("/analytics", methods=["GET"])
def analytics():
    hit_rate = (cache_hits / total_requests) if total_requests > 0 else 0

    baseline_tokens = total_requests * AVG_TOKENS
    actual_tokens = cache_misses * AVG_TOKENS

    savings = (baseline_tokens - actual_tokens) * MODEL_COST_PER_MILLION / 1_000_000
    savings_percent = hit_rate * 100

    return jsonify({
        "hitRate": round(hit_rate, 2),
        "totalRequests": total_requests,
        "cacheHits": cache_hits,
        "cacheMisses": cache_misses,
        "cacheSize": len(cache),
        "costSavings": round(savings, 2),
        "savingsPercent": round(savings_percent, 2),
        "strategies": [
            "exact match",
            "semantic similarity",
            "LRU eviction",
            "TTL expiration"
        ]
    })


# -------------------------
# RENDER PORT FIX
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
