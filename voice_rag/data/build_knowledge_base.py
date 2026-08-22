"""
Comprehensive Knowledge Base Generator & Index Builder.

Builds a high-density, rich multilingual knowledge corpus (Hindi + English)
across diverse domains (Goa, Indian History, Geography, Science, AI, Space,
Culture, Government) and indexes it into FAISS and BM25 using the
fine-tuned multilingual embedder.
"""

from __future__ import annotations

import json
from pathlib import Path
from loguru import logger

from voice_rag.config import get_settings
from voice_rag.indexing.build_bm25 import build_bm25_index
from voice_rag.indexing.build_faiss import build_faiss_index
from voice_rag.indexing.chunkers import chunk_text
from voice_rag.indexing.embeddings import Embedder
from voice_rag.pipeline.schemas import ChunkMetadata, ChunkStrategy, DocumentRecord

KNOWLEDGE_RECORDS = [
    # ── Goa & HH Goa ──────────────────────────────────────────────────────────
    {
        "query_id": "goa_01",
        "query": "Tell me about Goa and its capital",
        "query_type": "description",
        "answer": "Goa is a state on the southwestern coast of India within the Konkan region. Its capital is Panaji (Panjim) and its largest city is Vasco da Gama.",
        "passages": [
            {"text": "Goa is a state on the southwestern coast of India within the Konkan region, geographically separated from the Deccan highlands by the Western Ghats. It is surrounded by Maharashtra to the north and Karnataka to the east and south, with the Arabian Sea forming its western coast.", "is_selected": 1, "index": 0},
            {"text": "Panaji, also known as Panjim, is the capital of the Indian state of Goa and the headquarters of the North Goa district. It lies on the banks of the Mandovi river estuary.", "is_selected": 1, "index": 1},
            {"text": "Vasco da Gama is the largest city in Goa, located on the western tip of the Mormugao peninsula at the mouth of the Zuari River.", "is_selected": 0, "index": 2},
            {"text": "Official language of Goa is Konkani, written in the Devanagari script. Marathi is also widely spoken and used for official purposes.", "is_selected": 1, "index": 3},
            {"text": "Goa was a Portuguese territory for about 450 years until it was annexed by India in December 1961 through Operation Vijay.", "is_selected": 1, "index": 4},
        ],
        "translated_passages": [
            {"text": "गोवा भारत के दक्षिण-पश्चिमी तट पर कोंकण क्षेत्र में स्थित एक राज्य है। इसकी राजधानी पणजी है और सबसे बड़ा शहर वास्को द गामा है।", "is_selected": 1, "index": 0},
            {"text": "पणजी भारतीय राज्य गोवा की राजधानी है और यह मांडवी नदी के किनारे स्थित है।", "is_selected": 1, "index": 1},
            {"text": "गोवा की आधिकारिक भाषा कोंकणी है, जो देवनागरी लिपि में लिखी जाती है।", "is_selected": 1, "index": 2},
            {"text": "गोवा लगभग 450 वर्षों तक पुर्तगाली उपनिवेश रहा, जिसे दिसंबर 1961 में ऑपरेशन विजय द्वारा भारत में शामिल किया गया।", "is_selected": 1, "index": 3},
        ],
    },
    {
        "query_id": "goa_02",
        "query": "What are the famous tourist attractions and beaches in Goa?",
        "query_type": "description",
        "answer": "Famous attractions in Goa include Baga, Calangute, and Anjuna beaches, Dudhsagar Falls, and the Basilica of Bom Jesus.",
        "passages": [
            {"text": "North Goa is famous for its vibrant beaches including Baga Beach, Calangute Beach, Anjuna Beach, and Vagator Beach, known for nightlife, water sports, and beach shacks.", "is_selected": 1, "index": 0},
            {"text": "South Goa is renowned for tranquil, pristine beaches such as Palolem Beach, Colva Beach, Agonda Beach, and Benaulim Beach.", "is_selected": 1, "index": 1},
            {"text": "Dudhsagar Falls is a four-tiered waterfall located on the Mandovi River in Goa. It is among India's tallest waterfalls with a height of 310 meters (1017 feet).", "is_selected": 1, "index": 2},
            {"text": "The Basilica of Bom Jesus in Old Goa is a UNESCO World Heritage Site that holds the mortal remains of St. Francis Xavier.", "is_selected": 1, "index": 3},
            {"text": "Fort Aguada is a well-preserved seventeenth-century Portuguese fort and lighthouse standing on Sinquerim Beach overlooking the Arabian Sea.", "is_selected": 1, "index": 4},
        ],
        "translated_passages": [
            {"text": "उत्तरी गोवा अपने जीवंत समुद्र तटों जैसे बागा, कलंगूट, अंजुना और वागाटोर बीच के लिए प्रसिद्ध है।", "is_selected": 1, "index": 0},
            {"text": "दूधसागर जलप्रपात गोवा में मांडवी नदी पर स्थित एक चार-स्तरीय झरना है, जिसकी ऊंचाई 310 मीटर है।", "is_selected": 1, "index": 1},
            {"text": "ओल्ड गोवा में बेसिलिका ऑफ बॉम जीसस एक यूनेस्को विश्व धरोहर स्थल है जिसमें सेंट फ्रांसिस जेवियर के अवशेष हैं।", "is_selected": 1, "index": 2},
        ],
    },

    # ── Indian Geography & Governance ─────────────────────────────────────────
    {
        "query_id": "india_geo_01",
        "query": "What is the capital of India?",
        "query_type": "factoid",
        "answer": "New Delhi is the capital city of India.",
        "passages": [
            {"text": "New Delhi is the capital city of India. It serves as the seat of all three branches of the Government of India: executive (Rashtrapati Bhavan), legislative (Parliament House / Sansad Bhavan), and judiciary (Supreme Court of India).", "is_selected": 1, "index": 0},
            {"text": "Mumbai is the financial, commercial, and entertainment capital of India, and houses the Reserve Bank of India, Bombay Stock Exchange, and National Stock Exchange.", "is_selected": 0, "index": 1},
            {"text": "India is a country in South Asia, bounded by the Indian Ocean on the south, the Arabian Sea on the southwest, and the Bay of Bengal on the southeast.", "is_selected": 0, "index": 2},
            {"text": "Kolkata was the capital of British India until 1911, when King George V announced the shifting of the capital to Delhi.", "is_selected": 1, "index": 3},
        ],
        "translated_passages": [
            {"text": "नई दिल्ली भारत की राजधानी है। यह भारत सरकार की कार्यकारी, विधायी और न्यायिक शाखाओं का केंद्र है।", "is_selected": 1, "index": 0},
            {"text": "मुंबई भारत की वित्तीय राजधानी है जहां भारतीय रिजर्व बैंक और बॉम्बे स्टॉक एक्सचेंज स्थित हैं।", "is_selected": 0, "index": 1},
            {"text": "1911 तक कोलकाता ब्रिटिश भारत की राजधानी था, जिसके बाद राजधानी को दिल्ली स्थानांतरित किया गया।", "is_selected": 1, "index": 2},
        ],
    },
    {
        "query_id": "india_geo_02",
        "query": "How many states and union territories are there in India?",
        "query_type": "factoid",
        "answer": "India has 28 states and 8 Union Territories.",
        "passages": [
            {"text": "India is a federal democratic republic comprising 28 States and 8 Union Territories. Each state has its own elected state legislative assembly and chief minister.", "is_selected": 1, "index": 0},
            {"text": "The 8 Union Territories of India are: Andaman and Nicobar Islands, Chandigarh, Dadra and Nagar Haveli and Daman and Diu, Delhi (NCT), Jammu and Kashmir, Ladakh, Lakshadweep, and Puducherry.", "is_selected": 1, "index": 1},
            {"text": "Rajasthan is the largest state in India by area, covering 342,239 square kilometers, while Goa is the smallest state by area.", "is_selected": 1, "index": 2},
            {"text": "Uttar Pradesh is the most populous state in India, with over 240 million residents.", "is_selected": 1, "index": 3},
        ],
        "translated_passages": [
            {"text": "भारत में वर्तमान में 28 राज्य और 8 केंद्र शासित प्रदेश हैं। प्रत्येक राज्य की अपनी निर्वाचित विधानसभा और मुख्यमंत्री होते हैं।", "is_selected": 1, "index": 0},
            {"text": "क्षेत्रफल की दृष्टि से राजस्थान भारत का सबसे बड़ा राज्य है और गोवा सबसे छोटा राज्य है।", "is_selected": 1, "index": 1},
            {"text": "उत्तर प्रदेश 24 करोड़ से अधिक आबादी के साथ भारत का सबसे अधिक जनसंख्या वाला राज्य है।", "is_selected": 1, "index": 2},
        ],
    },

    # ── National Symbols of India ─────────────────────────────────────────────
    {
        "query_id": "symbols_01",
        "query": "What is the national bird and national animal of India?",
        "query_type": "factoid",
        "answer": "The national bird of India is the Indian Peacock (Pavo cristatus) and the national animal is the Royal Bengal Tiger (Panthera tigris).",
        "passages": [
            {"text": "The Indian Peacock (Pavo cristatus) was declared the National Bird of India in 1963 because of its rich religious and legendary involvement in Indian traditions and its universal beauty.", "is_selected": 1, "index": 0},
            {"text": "The Royal Bengal Tiger (Panthera tigris) was declared the National Animal of India in April 1973 with the initiation of Project Tiger to protect the endangered tiger population.", "is_selected": 1, "index": 1},
            {"text": "The National Flower of India is the Lotus (Nelumbo nucifera), which represents purity, auspiciousness, and spiritual enlightenment in Indian culture.", "is_selected": 1, "index": 2},
            {"text": "The National Tree of India is the Banyan Tree (Ficus benghalensis), symbolizing eternal life and immortality due to its ever-expanding branches and roots.", "is_selected": 1, "index": 3},
            {"text": "The National River of India is the Ganga (Ganges), the longest river in India sacred to Hindus.", "is_selected": 1, "index": 4},
        ],
        "translated_passages": [
            {"text": "भारत का राष्ट्रीय पक्षी भारतीय मोर (पावो क्रिस्टेटस) है, जिसे 1963 में राष्ट्रीय पक्षी घोषित किया गया था।", "is_selected": 1, "index": 0},
            {"text": "भारत का राष्ट्रीय पशु रॉयल बंगाल टाइगर (बाघ) है, जिसे 1973 में प्रोजेक्ट टाइगर के तहत राष्ट्रीय पशु घोषित किया गया।", "is_selected": 1, "index": 1},
            {"text": "भारत का राष्ट्रीय फूल कमल (लोटस) है जो पवित्रता और दिव्यता का प्रतीक है।", "is_selected": 1, "index": 2},
            {"text": "भारत का राष्ट्रीय वृक्ष बरगद का पेड़ है और राष्ट्रीय नदी गंगा है।", "is_selected": 1, "index": 3},
        ],
    },

    # ── Science, Biology & Space ──────────────────────────────────────────────
    {
        "query_id": "science_01",
        "query": "How does photosynthesis work?",
        "query_type": "description",
        "answer": "Photosynthesis is the process where green plants use sunlight, water, and carbon dioxide to create oxygen and glucose.",
        "passages": [
            {"text": "Photosynthesis is a biological process by which autotrophic organisms (like green plants and algae) convert solar light energy into chemical energy stored in glucose molecules. The chemical equation is 6CO2 + 6H2O + light energy -> C6H12O6 + 6O2.", "is_selected": 1, "index": 0},
            {"text": "Chlorophyll, the green pigment located inside chloroplasts in plant leaves, absorbs blue and red wavelengths of light while reflecting green light.", "is_selected": 1, "index": 1},
            {"text": "Photosynthesis occurs in two stages: the light-dependent reactions (occurring in thylakoids) and the light-independent Calvin cycle (occurring in the stroma).", "is_selected": 1, "index": 2},
            {"text": "As a vital byproduct of photosynthesis, oxygen is released into the Earth's atmosphere, enabling aerobic life on Earth.", "is_selected": 1, "index": 3},
        ],
        "translated_passages": [
            {"text": "प्रकाश संश्लेषण (Photosynthesis) वह प्रक्रिया है जिसके द्वारा हरे पौधे सूर्य के प्रकाश, जल और कार्बन डाइऑक्साइड का उपयोग करके ऑक्सीजन और ग्लूकोज का निर्माण करते हैं।", "is_selected": 1, "index": 0},
            {"text": "क्लोरोफिल (पर्णहरित) पत्तियों में पाया जाने वाला हरा वर्णक है जो सूर्य के प्रकाश को अवशोषित करता है।", "is_selected": 1, "index": 1},
        ],
    },
    {
        "query_id": "space_01",
        "query": "What are the major achievements of ISRO like Chandrayaan and Mangalyaan?",
        "query_type": "description",
        "answer": "ISRO's major achievements include Chandrayaan-3 landing on the Moon's South Pole, Mangalyaan reaching Mars orbit, and Aditya-L1 studying the Sun.",
        "passages": [
            {"text": "The Indian Space Research Organisation (ISRO) is India's national space agency, founded in 1969 by Dr. Vikram Sarabhai and headquartered in Bengaluru.", "is_selected": 1, "index": 0},
            {"text": "Chandrayaan-3 achieved a historic soft landing on the lunar South Pole on August 23, 2023, making India the first nation to land near the lunar south pole and fourth country to land on the Moon.", "is_selected": 1, "index": 1},
            {"text": "Mars Orbiter Mission (Mangalyaan), launched in 2013, successfully entered Mars orbit on its first attempt in September 2014, making ISRO the fourth space agency to reach Mars.", "is_selected": 1, "index": 2},
            {"text": "Aditya-L1 is India's first dedicated solar observatory spacecraft, positioned at the Sun-Earth Lagrangian point L1 to observe the solar corona and chromospheric dynamics.", "is_selected": 1, "index": 3},
            {"text": "Gaganyaan is ISRO's human spaceflight program aiming to demonstrate human spaceflight capability by launching astronauts to low Earth orbit.", "is_selected": 1, "index": 4},
        ],
        "translated_passages": [
            {"text": "भारतीय अंतरिक्ष अनुसंधान संगठन (ISRO) की स्थापना 1969 में डॉ. विक्रम साराभाई द्वारा की गई थी। इसका मुख्यालय बेंगलुरु में है।", "is_selected": 1, "index": 0},
            {"text": "चंद्रयान-3 ने 23 अगस्त 2023 को चंद्रमा के दक्षिणी ध्रुव पर ऐतिहासिक सॉफ्ट लैंडिंग की, जिससे भारत चंद्रमा के दक्षिणी ध्रुव पर उतरने वाला दुनिया का पहला देश बन गया।", "is_selected": 1, "index": 1},
            {"text": "मंगलयान (MOM) ने 2014 में अपने पहले ही प्रयास में मंगल ग्रह की कक्षा में सफलतापूर्वक प्रवेश किया था।", "is_selected": 1, "index": 2},
            {"text": "आदित्य-एल1 सूर्य का अध्ययन करने वाला भारत का पहला समर्पित सौर मिशन है।", "is_selected": 1, "index": 3},
        ],
    },

    # ── History & Monuments ───────────────────────────────────────────────────
    {
        "query_id": "history_01",
        "query": "Who built the Taj Mahal and where is it located?",
        "query_type": "factoid",
        "answer": "The Taj Mahal was built in Agra, Uttar Pradesh, by Mughal Emperor Shah Jahan in memory of his wife Mumtaz Mahal.",
        "passages": [
            {"text": "The Taj Mahal is an ivory-white marble mausoleum on the south bank of the Yamuna river in Agra, Uttar Pradesh, India. It was commissioned in 1631 by Mughal Emperor Shah Jahan to house the tomb of his favourite wife, Mumtaz Mahal.", "is_selected": 1, "index": 0},
            {"text": "Construction of the Taj Mahal was completed around 1648, employing over 20,000 artisans led by court architect Ustad Ahmad Lahori. It was designated as a UNESCO World Heritage Site in 1983.", "is_selected": 1, "index": 1},
            {"text": "The Taj Mahal is considered the finest example of Mughal architecture, combining elements from Islamic, Persian, Ottoman Turkish, and Indian architectural styles.", "is_selected": 1, "index": 2},
        ],
        "translated_passages": [
            {"text": "ताजमहल उत्तर प्रदेश के आगरा में यमुना नदी के तट पर स्थित सफेद संगमरमर का एक मकबरा है। इसे मुगल सम्राट शाहजहाँ ने 1631 में अपनी बेगम मुमताज महल की याद में बनवाया था।", "is_selected": 1, "index": 0},
            {"text": "ताजमहल को 1983 में यूनेस्को विश्व धरोहर स्थल घोषित किया गया और यह दुनिया के सात अजूबों में से एक है।", "is_selected": 1, "index": 1},
        ],
    },
    {
        "query_id": "history_02",
        "query": "Who is known as the Father of the Indian Constitution?",
        "query_type": "factoid",
        "answer": "Dr. B.R. Ambedkar is recognized as the Father of the Indian Constitution.",
        "passages": [
            {"text": "Dr. Bhimrao Ramji Ambedkar served as the Chairman of the Drafting Committee of the Constituent Assembly and is widely revered as the Father of the Constitution of India.", "is_selected": 1, "index": 0},
            {"text": "The Constitution of India was adopted by the Constituent Assembly on 26 November 1949 and came into effect on 26 January 1950, celebrated nationwide as Republic Day.", "is_selected": 1, "index": 1},
            {"text": "The Indian Constitution is the longest written national constitution in the world, establishing India as a sovereign, socialist, secular, democratic republic.", "is_selected": 1, "index": 2},
        ],
        "translated_passages": [
            {"text": "डॉ. भीमराव रामजी अंबेडकर को भारतीय संविधान का जनक (Father of Indian Constitution) माना जाता है। वे संविधान प्रारूप समिति के अध्यक्ष थे।", "is_selected": 1, "index": 0},
            {"text": "भारतीय संविधान 26 नवंबर 1949 को अपनाया गया और 26 जनवरी 1950 को लागू हुआ, जिसे हम गणतंत्र दिवस के रूप में मनाते हैं।", "is_selected": 1, "index": 1},
        ],
    },

    # ── Artificial Intelligence, RAG & Vector Databases ──────────────────────
    {
        "query_id": "tech_01",
        "query": "What is Retrieval-Augmented Generation (RAG) and how does FAISS work?",
        "query_type": "description",
        "answer": "RAG enhances LLMs by retrieving relevant factual passages from external vector databases before generating grounded answers.",
        "passages": [
            {"text": "Retrieval-Augmented Generation (RAG) is an AI architecture that enhances Large Language Model (LLM) responses by querying an external knowledge base for relevant context passages and feeding them into the prompt.", "is_selected": 1, "index": 0},
            {"text": "RAG significantly mitigates LLM hallucinations, enables dynamic knowledge updating without retraining, and provides traceable source provenance through passage citations.", "is_selected": 1, "index": 1},
            {"text": "FAISS (Facebook AI Similarity Search) is a high-performance library for efficient similarity search and clustering of dense embedding vectors in multi-dimensional space, supporting IndexFlatIP, HNSW, and IVF algorithms.", "is_selected": 1, "index": 2},
            {"text": "BM25 (Best Matching 25) is a ranking function used in information retrieval that scores documents based on query term frequency and inverse document frequency, complementing dense vector search in hybrid retrieval architectures.", "is_selected": 1, "index": 3},
            {"text": "Reciprocal Rank Fusion (RRF) and Cross-Encoder rerankers combine dense vector and lexical BM25 results into a unified high-precision candidate set.", "is_selected": 1, "index": 4},
        ],
        "translated_passages": [
            {"text": "रिट्रीवल-ऑगमेंटेड जेनरेशन (RAG) एक ऐसी तकनीक है जो बाहरी नॉलेज बेस से प्रासंगिक डेटा लाकर एलएलएम (LLM) को सटीक और तथ्यपूर्ण उत्तर देने में सक्षम बनाती है।", "is_selected": 1, "index": 0},
            {"text": "FAISS (फेसबुक एआई सिमिलरिटी सर्च) घने एम्बेडिंग वैक्टर की तीव्र खोज और तुलना के लिए एक अत्यधिक अनुकूलित लाइब्रेरी है।", "is_selected": 1, "index": 1},
            {"text": "BM25 लेक्सिकल सर्च और FAISS डेंस सर्च को हाइब्रिड फ्यूजन और क्रॉस-एनकोडर रीरैंकिंग द्वारा मिलाकर उच्च सटीकता प्राप्त की जाती है।", "is_selected": 1, "index": 2},
        ],
    },
    {
        "query_id": "tech_02",
        "query": "What are Sarvam AI and ElevenLabs speech models?",
        "query_type": "description",
        "answer": "Sarvam AI builds generative AI and speech models (Saaras, Bulbul, Sarvam 105B) for Indic languages, while ElevenLabs provides expressive multilingual voice AI.",
        "passages": [
            {"text": "Sarvam AI is an Indian artificial intelligence startup building full-stack foundational AI models specialized for Indian languages, including Saaras STT (Speech-to-Text), Bulbul TTS (Text-to-Speech), and Sarvam-105B LLM.", "is_selected": 1, "index": 0},
            {"text": "Sarvam's Saaras speech model is trained specifically on Indian accents, dialects, and mixed code-switching (Hinglish, Tanglish, etc.), delivering superior transcription accuracy for Indic speech.", "is_selected": 1, "index": 1},
            {"text": "ElevenLabs is a voice AI technology company developing advanced neural speech synthesis (eleven_multilingual_v2) and automatic speech recognition (Scribe v1) across dozens of languages.", "is_selected": 1, "index": 2},
        ],
        "translated_passages": [
            {"text": "सर्वम एआई (Sarvam AI) भारतीय भाषाओं के लिए विशेष रूप से विकसित स्पीच (Saaras, Bulbul) और लैंग्वेज मॉडल (Sarvam 105B) प्रदान करता है।", "is_selected": 1, "index": 0},
            {"text": "इलेवनलैब्स (ElevenLabs) उच्च गुणवत्ता वाले बहुभाषी वॉइस सिंथेसिस और स्पीच ट्रांसक्रिप्शन मॉडल विकसित करता है।", "is_selected": 1, "index": 1},
        ],
    },
]


def build_full_knowledge_base():
    """Build FAISS and BM25 database from the full knowledge base."""
    cfg = get_settings()
    all_chunks: list[ChunkMetadata] = []

    logger.info(f"Extracting passages from {len(KNOWLEDGE_RECORDS)} knowledge records...")

    for rec in KNOWLEDGE_RECORDS:
        doc = DocumentRecord(
            query_id=rec["query_id"],
            query=rec["query"],
            query_type=rec["query_type"],
            answer=rec["answer"],
            passages=rec["passages"],
            translated_passages=rec.get("translated_passages", []),
            language="multilingual",
        )

        # Chunk English passages
        for p in rec["passages"]:
            p_text = p.get("text", "")
            if p_text:
                chunks = chunk_text(
                    text=p_text,
                    document_id=doc.query_id,
                    strategy=ChunkStrategy.ADAPTIVE,
                    language="en",
                    is_selected=bool(p.get("is_selected", 0)),
                )
                all_chunks.extend(chunks)

        # Chunk Hindi / Indic translated passages
        for tp in rec.get("translated_passages", []):
            tp_text = tp.get("text", "")
            if tp_text:
                chunks = chunk_text(
                    text=tp_text,
                    document_id=f"{doc.query_id}_hi",
                    strategy=ChunkStrategy.ADAPTIVE,
                    language="hi",
                    is_selected=bool(tp.get("is_selected", 0)),
                )
                all_chunks.extend(chunks)

    logger.info(f"Generated {len(all_chunks)} multilingual chunk records.")

    # Use fine-tuned local embedder if available, otherwise default
    local_embedder_path = Path("finetuned_multilingual_embedder")
    if local_embedder_path.exists():
        logger.info(f"Using fine-tuned multilingual embedder from {local_embedder_path}")
        embedder = Embedder(model_name=str(local_embedder_path))
    else:
        embedder = Embedder()

    # 1. Build FAISS index
    logger.info("Building FAISS Dense Index...")
    faiss_idx, ordered_chunks = build_faiss_index(
        chunks=all_chunks,
        embedder=embedder,
        index_dir=cfg.index_dir,
        batch_size=64,
        show_progress=True,
    )
    logger.info(f"FAISS index built successfully: {faiss_idx.ntotal} vectors.")

    # 2. Build BM25 index
    logger.info("Building BM25 Lexical Index...")
    bm25_idx, bm25_chunks = build_bm25_index(
        chunks=all_chunks,
        index_dir=cfg.index_dir,
    )
    logger.info(f"BM25 index built successfully: {bm25_idx.corpus_size} documents.")

    return len(all_chunks)


if __name__ == "__main__":
    count = build_full_knowledge_base()
    print(f"Complete Knowledge Base built and indexed with {count} chunks!")
