"""
Grounded prompt templates for the RAG generation pipeline.

All prompts enforce strict anti-hallucination rules:
  - Answer ONLY from the provided context passages
  - Cite passage numbers
  - Refuse if evidence is insufficient
"""

from __future__ import annotations

from typing import Sequence

from voice_rag.pipeline.schemas import RetrievedChunk


# ═══════════════════════════════════════════════════════════════════════════
# System prompt
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a precise, grounded question-answering assistant.

## RULES — follow these exactly:
1. Answer ONLY using information from the CONTEXT PASSAGES provided below.
2. If the passages do not contain enough information to answer, respond with:
   "I don't have enough information in the provided knowledge base to answer this question."
3. NEVER fabricate, invent, or infer facts beyond what the passages explicitly state.
4. Cite the passage number(s) you used, e.g. [Passage 1], [Passage 3].
5. Keep answers concise, accurate, and well-structured.
6. If the question is in Hindi or another Indic language, answer in that same language.
7. If passages are in multiple languages, prefer the language of the question.
"""


# ═══════════════════════════════════════════════════════════════════════════
# Context builder
# ═══════════════════════════════════════════════════════════════════════════

def build_context_block(chunks: Sequence[RetrievedChunk], max_passages: int = 5) -> str:
    """
    Format retrieved chunks into a numbered context block for the LLM.

    Args:
        chunks:        Ranked retrieved chunks.
        max_passages:  Maximum number of passages to include.

    Returns:
        Formatted context string.
    """
    lines: list[str] = ["## CONTEXT PASSAGES\n"]
    for i, rc in enumerate(chunks[:max_passages]):
        score_info = f"(relevance: {rc.rerank_score:.3f})" if rc.rerank_score else ""
        lines.append(f"### Passage {i + 1} {score_info}")
        lines.append(rc.chunk.text)
        lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# User prompt builder
# ═══════════════════════════════════════════════════════════════════════════

def build_user_prompt(query: str, chunks: Sequence[RetrievedChunk]) -> str:
    """
    Build the full user prompt with context and question.

    Args:
        query:  The user's question.
        chunks: Ranked retrieved chunks.

    Returns:
        Complete user prompt string.
    """
    context = build_context_block(chunks)
    return f"""{context}

## QUESTION
{query}

## INSTRUCTIONS
Answer the question using ONLY the context passages above. Cite passage numbers. If the context is insufficient, say so explicitly.
"""


# ═══════════════════════════════════════════════════════════════════════════
# Refusal templates
# ═══════════════════════════════════════════════════════════════════════════

REFUSAL_NO_EVIDENCE = (
    "I don't have enough information in the provided knowledge base "
    "to answer this question."
)

REFUSAL_SAFETY = (
    "I'm unable to respond to this request as it may involve "
    "harmful or inappropriate content."
)

REFUSAL_OFF_TOPIC = (
    "This question appears to be outside the scope of the knowledge base. "
    "I can only answer questions related to the indexed content."
)
