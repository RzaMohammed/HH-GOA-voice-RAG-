"""
Streaming loader for the ai4bharat/MSMARCO-XI dataset.

Handles dynamic schema inspection — the HF dataset has varying field names
across splits/versions (e.g. ``Translated_passages`` vs ``passages``).
Includes a built-in local sample generator for offline/CI usage.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Iterator, Optional

from loguru import logger

from voice_rag.config import get_settings
from voice_rag.pipeline.schemas import DocumentRecord


# ═══════════════════════════════════════════════════════════════════════════
# Field mapping — handles schema variants across MSMARCO-XI versions
# ═══════════════════════════════════════════════════════════════════════════

_FIELD_MAP = {
    "query": ["query", "Query"],
    "query_id": ["query_id", "QueryId", "id"],
    "query_type": ["query_type", "QueryType"],
    "answer": ["Answer", "answer", "wellFormedAnswers"],
    "passages": ["passages", "Passages"],
    "translated_passages": ["Translated_passages", "translated_passages"],
    "english_passages": ["English_passages", "english_passages"],
    "is_selected": ["is_selected", "IsSelected"],
}


def _resolve_field(row: dict, field_key: str, default=None):
    """Try multiple possible column names for a logical field."""
    for candidate in _FIELD_MAP.get(field_key, [field_key]):
        if candidate in row:
            return row[candidate]
    return default


# ═══════════════════════════════════════════════════════════════════════════
# Passage extraction helper
# ═══════════════════════════════════════════════════════════════════════════

def _extract_passages(row: dict) -> tuple[list[dict], list[dict]]:
    """
    Extract passage dicts and translated passage dicts from a row.

    MSMARCO-XI stores passages either as:
    - ``passages`` dict with keys ``passage_text`` and ``is_selected``
    - ``Translated_passages`` and ``English_passages`` with parallel arrays
    """
    passages: list[dict] = []
    translated: list[dict] = []

    # --- Standard passages dict ---
    raw_passages = _resolve_field(row, "passages")
    if isinstance(raw_passages, dict):
        texts = raw_passages.get("passage_text", [])
        selected = raw_passages.get("is_selected", [0] * len(texts))
        for i, (txt, sel) in enumerate(zip(texts, selected)):
            passages.append({
                "text": str(txt),
                "is_selected": int(sel),
                "index": i,
            })
    elif isinstance(raw_passages, list):
        for i, p in enumerate(raw_passages):
            if isinstance(p, dict):
                passages.append({
                    "text": p.get("passage_text", p.get("text", "")),
                    "is_selected": int(p.get("is_selected", 0)),
                    "index": i,
                })
            else:
                passages.append({"text": str(p), "is_selected": 0, "index": i})

    # --- Translated passages ---
    raw_translated = _resolve_field(row, "translated_passages")
    if isinstance(raw_translated, dict):
        texts = raw_translated.get("passage_text", [])
        selected = raw_translated.get("is_selected", [0] * len(texts))
        for i, (txt, sel) in enumerate(zip(texts, selected)):
            translated.append({
                "text": str(txt),
                "is_selected": int(sel),
                "index": i,
            })
    elif isinstance(raw_translated, list):
        for i, p in enumerate(raw_translated):
            if isinstance(p, dict):
                translated.append({
                    "text": p.get("passage_text", p.get("text", "")),
                    "is_selected": int(p.get("is_selected", 0)),
                    "index": i,
                })

    # --- English passages (fallback) ---
    if not passages:
        raw_english = _resolve_field(row, "english_passages")
        if isinstance(raw_english, dict):
            texts = raw_english.get("passage_text", [])
            selected = raw_english.get("is_selected", [0] * len(texts))
            for i, (txt, sel) in enumerate(zip(texts, selected)):
                passages.append({
                    "text": str(txt),
                    "is_selected": int(sel),
                    "index": i,
                })

    return passages, translated


# ═══════════════════════════════════════════════════════════════════════════
# Row → DocumentRecord
# ═══════════════════════════════════════════════════════════════════════════

def _row_to_record(row: dict, language: str = "en") -> DocumentRecord:
    """Convert a raw HF dataset row to a typed DocumentRecord."""
    passages, translated = _extract_passages(row)
    return DocumentRecord(
        query_id=str(_resolve_field(row, "query_id", "")),
        query=str(_resolve_field(row, "query", "")),
        query_type=str(_resolve_field(row, "query_type", "")),
        answer=str(_resolve_field(row, "answer", "")),
        passages=passages,
        translated_passages=translated,
        language=language,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Streaming loader
# ═══════════════════════════════════════════════════════════════════════════

def stream_dataset(
    max_samples: Optional[int] = None,
    language: Optional[str] = None,
    split: Optional[str] = None,
) -> Iterator[DocumentRecord]:
    """
    Stream MSMARCO-XI records from HuggingFace.

    Falls back to the built-in local sample if the HF dataset cannot
    be loaded (no network, missing auth, etc.).
    """
    cfg = get_settings()
    max_samples = max_samples or cfg.dataset_max_samples
    language = language or cfg.dataset_language
    split = split or cfg.dataset_split

    try:
        from datasets import load_dataset

        logger.info(
            f"Streaming {cfg.dataset_name} | lang={language} | split={split} | max={max_samples}"
        )
        ds = load_dataset(
            cfg.dataset_name,
            language,
            split=split,
            streaming=cfg.dataset_streaming,
            trust_remote_code=True,
        )

        count = 0
        for row in ds:
            if max_samples and count >= max_samples:
                break
            yield _row_to_record(dict(row), language=language)
            count += 1

        logger.info(f"Streamed {count} records from {cfg.dataset_name}")

    except Exception as exc:
        logger.warning(f"HF dataset load failed ({exc}), falling back to local sample")
        yield from load_local_sample(max_samples=max_samples)


# ═══════════════════════════════════════════════════════════════════════════
# Local sample for offline / CI usage
# ═══════════════════════════════════════════════════════════════════════════

_BUILTIN_SAMPLES = [
    {
        "query_id": "sample_001",
        "query": "What is the capital of India?",
        "query_type": "factoid",
        "answer": "New Delhi is the capital of India.",
        "passages": [
            {"text": "New Delhi is the capital city of India. It serves as the seat of the Government of India.", "is_selected": 1, "index": 0},
            {"text": "Mumbai is the financial capital of India and the most populous city.", "is_selected": 0, "index": 1},
            {"text": "India is a country in South Asia. It is the seventh-largest country by area.", "is_selected": 0, "index": 2},
            {"text": "Delhi, officially the National Capital Territory of Delhi, is a city and union territory.", "is_selected": 1, "index": 3},
            {"text": "Kolkata is the capital of West Bengal and a major metropolitan city.", "is_selected": 0, "index": 4},
        ],
        "translated_passages": [
            {"text": "नई दिल्ली भारत की राजधानी है। यह भारत सरकार की सीट के रूप में कार्य करती है।", "is_selected": 1, "index": 0},
            {"text": "मुंबई भारत की वित्तीय राजधानी और सबसे अधिक आबादी वाला शहर है।", "is_selected": 0, "index": 1},
            {"text": "भारत दक्षिण एशिया का एक देश है। क्षेत्रफल के हिसाब से यह सातवां सबसे बड़ा देश है।", "is_selected": 0, "index": 2},
            {"text": "दिल्ली, आधिकारिक तौर पर राष्ट्रीय राजधानी क्षेत्र दिल्ली, एक शहर और केंद्र शासित प्रदेश है।", "is_selected": 1, "index": 3},
            {"text": "कोलकाता पश्चिम बंगाल की राजधानी और एक प्रमुख महानगरीय शहर है।", "is_selected": 0, "index": 4},
        ],
    },
    {
        "query_id": "sample_002",
        "query": "How does photosynthesis work?",
        "query_type": "description",
        "answer": "Photosynthesis is the process by which plants convert light energy into chemical energy.",
        "passages": [
            {"text": "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy that can be stored and later released to fuel the organism's activities.", "is_selected": 1, "index": 0},
            {"text": "The process of photosynthesis occurs in two main stages: the light-dependent reactions and the Calvin cycle.", "is_selected": 1, "index": 1},
            {"text": "Chlorophyll is the green pigment found in plants that absorbs sunlight for photosynthesis.", "is_selected": 1, "index": 2},
            {"text": "Cellular respiration is the process by which cells break down glucose to produce ATP.", "is_selected": 0, "index": 3},
            {"text": "Plants require water, carbon dioxide, and sunlight to perform photosynthesis and produce oxygen as a byproduct.", "is_selected": 1, "index": 4},
        ],
        "translated_passages": [],
    },
    {
        "query_id": "sample_003",
        "query": "भारत में कितने राज्य हैं?",
        "query_type": "factoid",
        "answer": "भारत में 28 राज्य और 8 केंद्र शासित प्रदेश हैं।",
        "passages": [
            {"text": "India is a federal union comprising 28 states and 8 union territories.", "is_selected": 1, "index": 0},
            {"text": "The states and union territories of India are sub-national administrative units.", "is_selected": 0, "index": 1},
            {"text": "Each state has its own elected government while union territories are ruled directly by the federal government.", "is_selected": 0, "index": 2},
        ],
        "translated_passages": [
            {"text": "भारत एक संघीय संघ है जिसमें 28 राज्य और 8 केंद्र शासित प्रदेश शामिल हैं।", "is_selected": 1, "index": 0},
            {"text": "भारत के राज्य और केंद्र शासित प्रदेश उप-राष्ट्रीय प्रशासनिक इकाइयाँ हैं।", "is_selected": 0, "index": 1},
            {"text": "प्रत्येक राज्य की अपनी निर्वाचित सरकार होती है जबकि केंद्र शासित प्रदेश सीधे संघीय सरकार द्वारा शासित होते हैं।", "is_selected": 0, "index": 2},
        ],
    },
    {
        "query_id": "sample_004",
        "query": "What are the benefits of machine learning?",
        "query_type": "description",
        "answer": "Machine learning enables automated pattern recognition, prediction, and decision-making from data.",
        "passages": [
            {"text": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.", "is_selected": 1, "index": 0},
            {"text": "Benefits of machine learning include automated decision making, pattern recognition in large datasets, predictive analytics, and natural language processing capabilities.", "is_selected": 1, "index": 1},
            {"text": "Deep learning is a type of machine learning based on artificial neural networks with representation learning.", "is_selected": 0, "index": 2},
            {"text": "Machine learning algorithms can process vast amounts of data quickly, identifying trends and patterns that would be impossible for humans to detect manually.", "is_selected": 1, "index": 3},
            {"text": "Cloud computing provides on-demand availability of computer system resources.", "is_selected": 0, "index": 4},
        ],
        "translated_passages": [],
    },
    {
        "query_id": "sample_005",
        "query": "What is the Taj Mahal?",
        "query_type": "description",
        "answer": "The Taj Mahal is an ivory-white marble mausoleum in Agra, India, built by Mughal emperor Shah Jahan.",
        "passages": [
            {"text": "The Taj Mahal is an ivory-white marble mausoleum on the right bank of the river Yamuna in Agra, Uttar Pradesh, India.", "is_selected": 1, "index": 0},
            {"text": "It was commissioned in 1631 by the fifth Mughal emperor, Shah Jahan, to house the tomb of his favourite wife, Mumtaz Mahal.", "is_selected": 1, "index": 1},
            {"text": "The Taj Mahal is regarded as one of the eight wonders of the world and is a UNESCO World Heritage Site.", "is_selected": 1, "index": 2},
            {"text": "The Red Fort is a historic fort in Delhi that served as the main residence of the Mughal Emperors.", "is_selected": 0, "index": 3},
            {"text": "Agra is a city on the banks of the Yamuna river in the Indian state of Uttar Pradesh.", "is_selected": 0, "index": 4},
        ],
        "translated_passages": [
            {"text": "ताज महल आगरा, उत्तर प्रदेश, भारत में यमुना नदी के दाहिने किनारे पर एक हाथीदांत-सफेद संगमरमर का मकबरा है।", "is_selected": 1, "index": 0},
            {"text": "इसे 1631 में पांचवें मुगल सम्राट शाहजहां ने अपनी पसंदीदा पत्नी मुमताज महल की कब्र रखने के लिए बनवाया था।", "is_selected": 1, "index": 1},
        ],
    },
]


def load_local_sample(max_samples: Optional[int] = None) -> Iterator[DocumentRecord]:
    """Yield built-in sample records for offline testing."""
    for i, raw in enumerate(_BUILTIN_SAMPLES):
        if max_samples and i >= max_samples:
            break
        yield DocumentRecord(
            query_id=raw["query_id"],
            query=raw["query"],
            query_type=raw["query_type"],
            answer=raw["answer"],
            passages=raw["passages"],
            translated_passages=raw.get("translated_passages", []),
            language="multilingual",
        )


def save_sample_cache(records: list[DocumentRecord], path: Optional[Path] = None) -> Path:
    """Persist a list of DocumentRecords to a JSON-Lines file."""
    cfg = get_settings()
    path = path or (cfg.data_dir / "msmarco_xi_sample.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")
    logger.info(f"Saved {len(records)} records to {path}")
    return path


def load_sample_cache(path: Optional[Path] = None) -> Iterator[DocumentRecord]:
    """Load records from a cached JSON-Lines file."""
    cfg = get_settings()
    path = path or (cfg.data_dir / "msmarco_xi_sample.jsonl")
    if not path.exists():
        logger.warning(f"Cache file {path} not found")
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield DocumentRecord.model_validate_json(line)
