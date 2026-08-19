# Voice-Enabled RAG System

> **HH Goa 2026 — Shortlisting Task 2**

A production-grade, voice-enabled Retrieval-Augmented Generation system with multilingual support, hybrid retrieval, grounding verification, and real-time pipeline telemetry.

## 🏗️ Architecture

```
Voice Input (Sarvam AI / ElevenLabs STT)
    → Query Guardrails (Safety + Relevance)
    → Parallel Hybrid Retrieval (FAISS Dense + BM25 Lexical)
    → Score Fusion (RRF / Weighted)
    → Cross-Encoder Reranking
    → Confidence Gating
    → Context-Grounded LLM Generation (Gemini / Groq / OpenAI)
    → Grounding / Hallucination Verification
    → Structured Response + Telemetry (P50/P70/P100)
```

## ✨ Key Features

- **Multi-Strategy Chunking**: Fixed, Sentence-aware, Semantic, and Adaptive chunking engines
- **Hybrid Retrieval**: FAISS dense search + BM25 lexical search with RRF/Weighted fusion
- **Cross-Encoder Reranking**: High-precision candidate reranking
- **Multi-Provider LLM**: Gemini, Groq, OpenAI with automatic fallback to mock
- **Voice STT**: Sarvam AI (Indic languages) + ElevenLabs with mock for offline testing
- **4-Layer Guardrails**: Safety, Relevance, Confidence, and Grounding verification
- **Pipeline Telemetry**: Per-stage latency tracking with P50/P70/P100 benchmarking
- **Voice-First UI**: Modern glassmorphism frontend with waveform visualization

## 📁 Project Structure

```
voice_rag/
├── config.py                 # Centralized settings
├── data/
│   ├── download.py           # MSMARCO-XI streaming loader
│   └── preprocessing.py      # Indic + English text cleaning
├── indexing/
│   ├── chunkers.py           # 4 chunking strategies
│   ├── embeddings.py         # Multilingual embedding model
│   ├── build_faiss.py        # FAISS index builder
│   └── build_bm25.py         # BM25 index builder
├── retrieval/
│   ├── dense.py              # FAISS dense search
│   ├── lexical.py            # BM25 lexical search
│   ├── hybrid.py             # RRF & Weighted Score Fusion
│   └── reranker.py           # Cross-encoder reranker
├── generation/
│   ├── prompts.py            # Anti-hallucination prompt templates
│   ├── llm.py                # Multi-provider LLM interface
│   └── grounding.py          # Faithfulness verifier
├── guardrails/
│   ├── safety.py             # Prompt injection & harmful content
│   ├── relevance.py          # Off-topic query detector
│   └── confidence.py         # Retrieval confidence gating
├── voice/
│   ├── sarvam_stt.py         # Sarvam AI STT client
│   └── elevenlabs_stt.py     # ElevenLabs STT client
├── pipeline/
│   ├── schemas.py            # Pydantic data models
│   └── orchestrator.py       # Async pipeline orchestrator
├── evaluation/
│   ├── retrieval_eval.py     # Recall@K, MRR evaluation
│   └── latency_eval.py       # P50/P70/P100 benchmark harness
└── api/
    └── main.py               # FastAPI REST + WebSocket server
frontend/
├── index.html                # Voice-first UI
├── style.css                 # Premium dark-mode design
└── app.js                    # Audio, API client, visualizations
tests/
├── test_chunkers.py
├── test_retrieval.py
├── test_guardrails.py
├── test_orchestrator.py
└── test_latency_eval.py
```

## 🚀 Quick Start

### 1. Install

```bash
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your API keys (all optional — system works offline with mocks)
```

### 3. Build Indices (optional — requires embedding model download)

```python
from voice_rag.data.download import load_local_sample
from voice_rag.indexing.chunkers import chunk_text
from voice_rag.indexing.build_faiss import build_faiss_index
from voice_rag.indexing.build_bm25 import build_bm25_index

# Get sample data
records = list(load_local_sample())
chunks = []
for rec in records:
    for p in rec.passages:
        chunks.extend(chunk_text(p["text"], document_id=rec.query_id))

# Build indices
build_faiss_index(chunks)
build_bm25_index(chunks)
```

### 4. Run Server

```bash
python -m voice_rag.api.main
# Open http://localhost:8000
```

### 5. Run Tests

```bash
pytest tests/ -v
```

### 6. Run Latency Benchmark

```bash
python -m voice_rag.evaluation.latency_eval
```

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `GROQ_API_KEY` | — | Groq API key |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `SARVAM_API_KEY` | — | Sarvam AI STT key |
| `ELEVENLABS_API_KEY` | — | ElevenLabs STT key |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Sentence transformer model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `LLM_PROVIDER` | `mock` | `gemini` / `groq` / `openai` / `mock` |
| `DEFAULT_CHUNK_STRATEGY` | `adaptive` | `fixed` / `sentence` / `semantic` / `adaptive` |
| `HYBRID_ALPHA` | `0.6` | Dense weight in fusion (0-1) |

## 📊 Evaluation Metrics

- **Recall@1, Recall@5, Recall@10** — retrieval coverage
- **MRR** — Mean Reciprocal Rank
- **P50, P70, P100** — latency percentiles
- **Grounding confidence** — claim-level faithfulness score

## License

MIT
