# Graph RAG Architecture for Azerbaijani Agricultural Sector

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE LAYER                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  Web App    │  │  REST API   │  │  CLI Tool   │  │  Slack/Telegram Bot │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
└─────────┼────────────────┼────────────────┼────────────────────┼────────────┘
          └────────────────┴────────────────┴────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ORCHESTRATION LAYER                                 │
│                              (LangGraph)                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        Query Processing Pipeline                      │   │
│  │  ┌─────────┐  ┌─────────────┐  ┌──────────┐  ┌────────────────────┐  │   │
│  │  │ Query   │→ │ Entity      │→ │ Query    │→ │ Retrieval Strategy │  │   │
│  │  │ Parser  │  │ Extraction  │  │ Classifier│  │ Router             │  │   │
│  │  └─────────┘  └─────────────┘  └──────────┘  └────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          HYBRID RETRIEVAL LAYER                              │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────────┐ │
│  │   VECTOR SEARCH    │  │   GRAPH TRAVERSAL  │  │    RESULT FUSION       │ │
│  │   (Qdrant)         │  │   (Neo4j)          │  │                        │ │
│  │                    │  │                    │  │  ┌──────────────────┐  │ │
│  │  • Dense vectors   │  │  • Cypher queries  │  │  │ Weighted Ranking │  │ │
│  │  • Sparse vectors  │  │  • Path traversal  │  │  │ Context Assembly │  │ │
│  │  • Metadata filter │  │  • Subgraph extract│  │  │ Deduplication    │  │ │
│  │                    │  │                    │  │  └──────────────────┘  │ │
│  └─────────┬──────────┘  └─────────┬──────────┘  └───────────┬────────────┘ │
│            └────────────────┬──────┴─────────────────────────┘              │
└─────────────────────────────┼───────────────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          GENERATION LAYER                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                     Answer Generation Pipeline                          │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │ │
│  │  │ Context      │→ │ LLM          │→ │ Citation     │→ │ Confidence │  │ │
│  │  │ Formatting   │  │ Generation   │  │ Injection    │  │ Scoring    │  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                     Hallucination Detection                             │ │
│  │  • Self-consistency check  • Retrieval-answer alignment  • Fact verify │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Ingestion Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DOCUMENT INGESTION PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────────┘

     ┌─────────────┐
     │  PDF Files  │
     │  (Raw)      │
     └──────┬──────┘
            │
            ▼
┌─────────────────────┐
│   PDF EXTRACTION    │
│   ┌───────────────┐ │
│   │ PyMuPDF       │ │    ┌─────────────────────────────────────┐
│   │ (Fast text)   │ │───▶│ Extracted Content                   │
│   └───────────────┘ │    │ • Text blocks                       │
│   ┌───────────────┐ │    │ • Tables (structured)               │
│   │ Unstructured  │ │    │ • Figures (base64)                  │
│   │ (Complex docs)│ │    │ • Metadata (title, date, author)    │
│   └───────────────┘ │    └──────────────────┬──────────────────┘
│   ┌───────────────┐ │                       │
│   │ Tesseract OCR │ │                       │
│   │ (Scanned)     │ │                       │
│   └───────────────┘ │                       │
└─────────────────────┘                       │
                                              ▼
                              ┌───────────────────────────────┐
                              │      LANGUAGE DETECTION       │
                              │  ┌─────┐ ┌─────┐ ┌─────────┐  │
                              │  │ AZ  │ │ RU  │ │ EN      │  │
                              │  └─────┘ └─────┘ └─────────┘  │
                              └───────────────┬───────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
        ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
        │  ENTITY-AWARE     │   │  HIERARCHICAL     │   │  STANDARD         │
        │  CHUNKING         │   │  CHUNKING         │   │  CHUNKING         │
        │                   │   │                   │   │                   │
        │ • Respects entity │   │ • Document level  │   │ • 500 tokens      │
        │   boundaries      │   │ • Section level   │   │ • 50 overlap      │
        │ • Context windows │   │ • Paragraph level │   │ • Recursive split │
        └─────────┬─────────┘   └─────────┬─────────┘   └─────────┬─────────┘
                  └─────────────────┬─────┴─────────────────┬─────┘
                                    ▼                       ▼
                        ┌─────────────────────┐   ┌─────────────────────┐
                        │   EMBEDDING         │   │   ENTITY            │
                        │   (BGE-M3)          │   │   EXTRACTION        │
                        │                     │   │   (GLiNER + LLM)    │
                        └──────────┬──────────┘   └──────────┬──────────┘
                                   │                         │
                                   ▼                         ▼
                        ┌─────────────────────┐   ┌─────────────────────┐
                        │      QDRANT         │   │      NEO4J          │
                        │   Vector Database   │   │   Graph Database    │
                        └─────────────────────┘   └─────────────────────┘
```

---

## Knowledge Graph Schema

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DOMAIN ONTOLOGY                                      │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────┐
                              │  REGULATION │
                              │─────────────│
                              │ title       │
                              │ doc_number  │
                              │ effective   │
                              │ authority   │
                              └──────┬──────┘
                                     │ REGULATES
                                     │ AMENDS
                                     ▼
┌─────────────┐  SUBSIDIZED_BY  ┌─────────────┐  GROWN_IN   ┌─────────────┐
│   SUBSIDY   │◀────────────────│    CROP     │────────────▶│   REGION    │
│   PROGRAM   │                 │─────────────│             │─────────────│
│─────────────│                 │ name        │             │ name        │
│ name        │                 │ scientific  │             │ type        │
│ type        │                 │ crop_type   │             │ area_ha     │
│ amount/ha   │                 │ water_req   │             │ climate     │
│ eligibility │                 └──────┬──────┘             └──────┬──────┘
└──────┬──────┘                        │                           │
       │                    ┌──────────┼──────────┐                │
       │ ADMINISTERED_BY    │          │          │                │ CONTAINS
       ▼                    ▼          ▼          ▼                │ BORDERS
┌─────────────┐     ┌───────────┐ ┌─────────┐ ┌─────────────┐      ▼
│ORGANIZATION │     │ AFFECTED  │ │RESISTANT│ │ REQUIRES    │  ┌─────────┐
│─────────────│     │    BY     │ │   TO    │ │   INPUT     │  │ (self)  │
│ name        │     └─────┬─────┘ └────┬────┘ └──────┬──────┘  └─────────┘
│ type        │           │            │             │
│ jurisdiction│           ▼            │             ▼
└─────────────┘    ┌─────────────┐     │      ┌─────────────┐
                   │   DISEASE   │     │      │    INPUT    │
                   │─────────────│     │      │─────────────│
                   │ name        │◀────┘      │ name        │
                   │ pathogen    │            │ type        │
                   │ symptoms    │            │ app_rate    │
                   │ severity    │            └──────┬──────┘
                   └──────┬──────┘                   │
                          │ TREATED_BY               │
                          └──────────────────────────┘
                   ┌─────────────┐
                   │    PEST     │──── TREATED_BY ───▶ INPUT
                   │─────────────│
                   │ name        │
                   │ pest_type   │
                   │ damage_type │
                   └──────┬──────┘
                          │ OCCURS_IN
                          ▼
                   ┌─────────────┐
                   │   SEASON    │
                   │─────────────│
                   │ name        │
                   │ start_month │
                   │ end_month   │
                   └─────────────┘
                   ┌─────────────┐
                   │  CLIMATE    │
                   │   FACTOR    │──── AFFECTED_BY (from Crop)
                   │─────────────│
                   │ name        │
                   │ factor_type │
                   │ impact      │
                   └─────────────┘
```

---

## Query Processing Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         QUERY CLASSIFICATION                                 │
└─────────────────────────────────────────────────────────────────────────────┘

                         User Query
                             │
                             ▼
                   ┌─────────────────┐
                   │  Query Parser   │
                   └────────┬────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
    ┌───────────────┐ ┌───────────┐ ┌───────────────┐
    │ Entity        │ │ Intent    │ │ Complexity    │
    │ Extraction    │ │ Detection │ │ Assessment    │
    └───────┬───────┘ └─────┬─────┘ └───────┬───────┘
            └───────────────┼───────────────┘
                            ▼
                   ┌─────────────────┐
                   │ Query Classifier│
                   └────────┬────────┘
                            │
     ┌──────────┬───────────┼───────────┬──────────┬──────────┐
     ▼          ▼           ▼           ▼          ▼          ▼
┌─────────┐ ┌────────┐ ┌─────────┐ ┌─────────┐ ┌────────┐ ┌────────┐
│FACTUAL  │ │RELATION│ │MULTI-HOP│ │AGGREGATE│ │COMPARE │ │TEMPORAL│
│         │ │        │ │         │ │         │ │        │ │        │
│"What is │ │"Which  │ │"What    │ │"Total   │ │"Compare│ │"How has│
│ wheat   │ │ crops  │ │ inputs  │ │ area of │ │ yields │ │ policy │
│ yield?" │ │ grow in│ │ treat   │ │ cotton?"│ │ in X   │ │ changed│
│         │ │ Aran?" │ │ cotton  │ │         │ │ vs Y?" │ │ since  │
│         │ │        │ │ pests in│ │         │ │        │ │ 2020?" │
│         │ │        │ │ Aran?"  │ │         │ │        │ │        │
└────┬────┘ └───┬────┘ └────┬────┘ └────┬────┘ └───┬────┘ └───┬────┘
     │          │           │           │          │          │
     ▼          ▼           ▼           ▼          ▼          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       RETRIEVAL STRATEGY ROUTER                              │
│                                                                              │
│  Query Type        │ Vector Weight │ Graph Weight │ Strategy                │
│  ──────────────────┼───────────────┼──────────────┼─────────────────────── │
│  FACTUAL           │     70%       │     30%      │ Vector-primary         │
│  RELATIONAL        │     30%       │     70%      │ Graph-primary          │
│  MULTI-HOP         │     20%       │     80%      │ Graph traversal        │
│  AGGREGATION       │     10%       │     90%      │ Cypher aggregation     │
│  COMPARATIVE       │     40%       │     60%      │ Parallel + merge       │
│  TEMPORAL          │     30%       │     70%      │ Time-filtered graph    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Hybrid Retrieval Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      HYBRID RETRIEVAL ENGINE                                 │
└─────────────────────────────────────────────────────────────────────────────┘

                          Classified Query
                                │
           ┌────────────────────┴────────────────────┐
           ▼                                         ▼
┌─────────────────────────┐             ┌─────────────────────────┐
│     VECTOR RETRIEVAL    │             │     GRAPH RETRIEVAL     │
│         (Qdrant)        │             │        (Neo4j)          │
├─────────────────────────┤             ├─────────────────────────┤
│                         │             │                         │
│  1. Embed query (BGE-M3)│             │  1. Extract entities    │
│         │               │             │         │               │
│         ▼               │             │         ▼               │
│  2. Dense search        │             │  2. NL → Cypher         │
│     (cosine sim)        │             │     translation         │
│         │               │             │         │               │
│         ▼               │             │         ▼               │
│  3. Sparse search       │             │  3. Execute traversal   │
│     (BM25)              │             │     MATCH patterns      │
│         │               │             │         │               │
│         ▼               │             │         ▼               │
│  4. Metadata filter     │             │  4. Extract subgraph    │
│     (region, date)      │             │     (nodes + edges)     │
│         │               │             │         │               │
│         ▼               │             │         ▼               │
│  5. Top-K chunks        │             │  5. Serialize paths     │
│                         │             │                         │
└───────────┬─────────────┘             └───────────┬─────────────┘
            │                                       │
            └──────────────────┬────────────────────┘
                               ▼
                ┌─────────────────────────────┐
                │       RESULT FUSION         │
                │                             │
                │  ┌───────────────────────┐  │
                │  │ 1. Reciprocal Rank    │  │
                │  │    Fusion (RRF)       │  │
                │  └───────────┬───────────┘  │
                │              ▼              │
                │  ┌───────────────────────┐  │
                │  │ 2. Weighted Scoring   │  │
                │  │    (query-dependent)  │  │
                │  └───────────┬───────────┘  │
                │              ▼              │
                │  ┌───────────────────────┐  │
                │  │ 3. Deduplication      │  │
                │  │    & Reranking        │  │
                │  └───────────┬───────────┘  │
                │              ▼              │
                │  ┌───────────────────────┐  │
                │  │ 4. Context Window     │  │
                │  │    Assembly           │  │
                │  └───────────────────────┘  │
                └──────────────┬──────────────┘
                               │
                               ▼
                      Retrieved Context
                    + Graph Reasoning Paths
```

---

## Entity Extraction Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ENTITY & RELATION EXTRACTION                              │
└─────────────────────────────────────────────────────────────────────────────┘

              Document Chunks
                    │
                    ▼
         ┌─────────────────────┐
         │   ZERO-SHOT NER     │
         │   (GLiNER Multi)    │
         │                     │
         │  Entity Types:      │
         │  • Crop             │
         │  • Region           │
         │  • Organization     │
         │  • Disease          │
         │  • Pest             │
         │  • Input            │
         │  • SubsidyProgram   │
         │  • Regulation       │
         │  • Season           │
         │  • ClimateFactor    │
         │  • Quantity         │
         │  • Date             │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  ENTITY RESOLUTION  │
         │                     │
         │  ┌───────────────┐  │      ┌─────────────────────────┐
         │  │ Normalization │  │◀─────│ Dictionaries            │
         │  │ Dictionaries  │  │      │ • regions.json (10)     │
         │  └───────┬───────┘  │      │ • crops.json (50+)      │
         │          │          │      │ • organizations.json    │
         │          ▼          │      └─────────────────────────┘
         │  ┌───────────────┐  │
         │  │ Fuzzy Match   │  │
         │  │ (RapidFuzz)   │  │
         │  │ threshold=85  │  │
         │  └───────┬───────┘  │
         │          │          │
         │          ▼          │
         │  ┌───────────────┐  │
         │  │ Coreference   │  │
         │  │ Resolution    │  │
         │  └───────────────┘  │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │ RELATION EXTRACTION │
         │ (LLM-based)         │
         │                     │
         │  Prompt Template:   │
         │  ┌───────────────┐  │
         │  │ Given entities│  │
         │  │ [e1, e2, ...]│  │
         │  │ and text,     │  │
         │  │ extract rels: │  │
         │  │ (e1, REL, e2) │  │
         │  └───────────────┘  │
         │                     │
         │  Valid Relations:   │
         │  • GROWN_IN         │
         │  • SUBSIDIZED_BY    │
         │  • AFFECTED_BY      │
         │  • RESISTANT_TO     │
         │  • TREATED_BY       │
         │  • REGULATES        │
         │  • ADMINISTERED_BY  │
         │  • OCCURS_IN        │
         │  • REQUIRES_INPUT   │
         │  • AMENDS           │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  PROVENANCE ATTACH  │
         │                     │
         │  Each node/edge:    │
         │  • source_doc_id    │
         │  • page_number      │
         │  • chunk_id         │
         │  • confidence_score │
         │  • extraction_date  │
         └──────────┬──────────┘
                    │
                    ▼
              Load to Neo4j
```

---

## Answer Generation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ANSWER GENERATION                                       │
└─────────────────────────────────────────────────────────────────────────────┘

    Retrieved Context              Graph Paths
         │                              │
         └──────────────┬───────────────┘
                        ▼
              ┌─────────────────┐
              │ CONTEXT BUILDER │
              │                 │
              │ <context>       │
              │   <chunk id=1>  │
              │     ...text...  │
              │   </chunk>      │
              │   <graph_path>  │
              │     A→REL→B→... │
              │   </graph_path> │
              │ </context>      │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  LLM GENERATION │
              │ (Claude 3.5)    │
              │                 │
              │ System Prompt:  │
              │ "Answer using   │
              │  ONLY context.  │
              │  Cite sources.  │
              │  If unsure, say │
              │  so."           │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ CITATION INJECT │
              │                 │
              │ "Cotton yields  │
              │  increased by   │
              │  15% [DOC-3,    │
              │  p.24]"         │
              └────────┬────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌─────────────────┐       ┌─────────────────┐
│ HALLUCINATION   │       │   CONFIDENCE    │
│   DETECTION     │       │    SCORING      │
│                 │       │                 │
│ • Self-consist  │       │ • Retrieval     │
│   (3 samples)   │       │   coverage      │
│ • Claim-context │       │ • Citation      │
│   alignment     │       │   density       │
│ • Fact verify   │       │ • Consistency   │
│   against graph │       │   score         │
└────────┬────────┘       └────────┬────────┘
         │                         │
         └────────────┬────────────┘
                      ▼
            ┌─────────────────┐
            │  FINAL ANSWER   │
            │                 │
            │ • Answer text   │
            │ • Citations     │
            │ • Confidence    │
            │ • Graph explain │
            │ • Caveats       │
            └─────────────────┘
```

---

## Technology Stack

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TECHNOLOGY STACK                                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER              │ TECHNOLOGY              │ PURPOSE                      │
├────────────────────┼─────────────────────────┼──────────────────────────────┤
│ PDF Extraction     │ PyMuPDF (fitz)          │ Fast text extraction         │
│                    │ Unstructured.io         │ Complex layouts, tables      │
│                    │ Tesseract OCR           │ Scanned documents            │
├────────────────────┼─────────────────────────┼──────────────────────────────┤
│ Entity Extraction  │ GLiNER (multi-v2.1)     │ Zero-shot NER                │
│                    │ Claude 3.5 Sonnet       │ Relation extraction          │
│                    │ RapidFuzz               │ Fuzzy entity matching        │
├────────────────────┼─────────────────────────┼──────────────────────────────┤
│ Graph Database     │ Neo4j                   │ Knowledge graph storage      │
│                    │ APOC Library            │ Advanced graph algorithms    │
├────────────────────┼─────────────────────────┼──────────────────────────────┤
│ Vector Database    │ Qdrant                  │ Hybrid dense+sparse search   │
├────────────────────┼─────────────────────────┼──────────────────────────────┤
│ Embeddings         │ BGE-M3 (BAAI)           │ Multilingual (AZ/RU/EN)      │
├────────────────────┼─────────────────────────┼──────────────────────────────┤
│ LLM Generation     │ Claude 3.5 Sonnet       │ Primary generation           │
│                    │ GPT-4 Turbo             │ Alternative                  │
│                    │ Llama 3.1 70B           │ Self-hosted option           │
├────────────────────┼─────────────────────────┼──────────────────────────────┤
│ Orchestration      │ LangGraph               │ Stateful workflow management │
├────────────────────┼─────────────────────────┼──────────────────────────────┤
│ API Framework      │ FastAPI                 │ REST API                     │
├────────────────────┼─────────────────────────┼──────────────────────────────┤
│ Deployment         │ Docker Compose          │ Container orchestration      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       DEPLOYMENT (Docker Compose)                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              DOCKER HOST                                     │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │    NGINX        │  │   FASTAPI       │  │        WORKERS              │  │
│  │   (Reverse      │  │   (API Server)  │  │                             │  │
│  │    Proxy)       │  │                 │  │  ┌─────────┐  ┌─────────┐   │  │
│  │                 │  │  Port: 8000     │  │  │ Celery  │  │ Celery  │   │  │
│  │  Port: 80/443   │──▶                 │──▶  │ Worker  │  │ Worker  │   │  │
│  │                 │  │                 │  │  │   #1    │  │   #2    │   │  │
│  │  SSL Termination│  │                 │  │  └────┬────┘  └────┬────┘   │  │
│  └─────────────────┘  └─────────────────┘  └───────┴────────────┴────────┘  │
│                                                     │                        │
│                         ┌───────────────────────────┘                        │
│                         │                                                    │
│                         ▼                                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         DATA LAYER                                   │    │
│  │                                                                      │    │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │    │
│  │  │     NEO4J       │  │     QDRANT      │  │      REDIS          │  │    │
│  │  │                 │  │                 │  │                     │  │    │
│  │  │  Port: 7474     │  │  Port: 6333     │  │  Port: 6379         │  │    │
│  │  │  (HTTP)         │  │  (REST)         │  │  (Cache/Queue)      │  │    │
│  │  │  Port: 7687     │  │  Port: 6334     │  │                     │  │    │
│  │  │  (Bolt)         │  │  (gRPC)         │  │                     │  │    │
│  │  │                 │  │                 │  │                     │  │    │
│  │  │  Volume:        │  │  Volume:        │  │  Volume:            │  │    │
│  │  │  ./neo4j/data   │  │  ./qdrant/data  │  │  ./redis/data       │  │    │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                       MONITORING                                     │    │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │    │
│  │  │   PROMETHEUS    │  │    GRAFANA      │  │   JAEGER            │  │    │
│  │  │   Port: 9090    │  │   Port: 3000    │  │   Port: 16686       │  │    │
│  │  │   (Metrics)     │  │   (Dashboards)  │  │   (Tracing)         │  │    │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      END-TO-END DATA FLOW                                    │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────────────────────┐
                    │          INGESTION FLOW              │
                    └──────────────────────────────────────┘

  PDF Documents ──▶ Extract ──▶ Chunk ──▶ ┬──▶ Embed ──▶ Qdrant
                                          │
                                          └──▶ Extract Entities ──▶ Neo4j


                    ┌──────────────────────────────────────┐
                    │           QUERY FLOW                 │
                    └──────────────────────────────────────┘

  User Query ──▶ Parse ──▶ Classify ──▶ Route ──▶ ┬──▶ Vector Search
                                                  │
                                                  └──▶ Graph Traversal
                                                            │
                                                            ▼
                                                     Fuse Results
                                                            │
                                                            ▼
                                                     Generate Answer
                                                            │
                                                            ▼
                                                     Inject Citations
                                                            │
                                                            ▼
                                                     Verify Faithfulness
                                                            │
                                                            ▼
                                                     Return Response
```

---

## Graph Query Patterns

```cypher
-- Pattern 1: Entity Neighborhood
MATCH (c:Crop {name: 'Cotton'})-[r]-(related)
RETURN c, type(r), related

-- Pattern 2: Multi-Hop Path (Crops → Regions → Subsidies)
MATCH (r:Region {name: 'Aran'})<-[:GROWN_IN]-(c:Crop)-[:SUBSIDIZED_BY]->(s:SubsidyProgram)
RETURN c.name, s.name, s.amount_per_hectare

-- Pattern 3: Disease Treatment Chain
MATCH (d:Disease)-[:AFFECTS]->(c:Crop)
MATCH (d)-[:TREATED_BY]->(i:Input)
WHERE c.name = 'Cotton'
RETURN d.name, i.name, i.application_rate

-- Pattern 4: Temporal Subsidy Comparison
MATCH (c:Crop {name: 'Wheat'})-[s:SUBSIDIZED_BY]->(p:SubsidyProgram)
WHERE p.year IN [2022, 2023, 2024]
RETURN p.year, p.amount_per_hectare
ORDER BY p.year

-- Pattern 5: Regional Aggregation
MATCH (r:Region)<-[g:GROWN_IN]-(c:Crop)
RETURN r.name, count(c) AS crop_count, sum(g.area_hectares) AS total_area
ORDER BY total_area DESC

-- Pattern 6: Regulation Impact Path
MATCH (reg:Regulation)-[:REGULATES]->(c:Crop)-[:GROWN_IN]->(r:Region)
WHERE reg.effective_date > date('2023-01-01')
RETURN reg.title, collect(DISTINCT c.name), collect(DISTINCT r.name)
```

---

## Implementation Timeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      16-WEEK IMPLEMENTATION ROADMAP                          │
└─────────────────────────────────────────────────────────────────────────────┘

Week    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16
        │    │    │    │    │    │    │    │    │    │    │    │    │    │    │    │
        ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼
     ┌──────────────┐
     │  PHASE 1:    │  • Environment setup
     │  Foundation  │  • Basic PDF processing
     │  (Weeks 1-3) │  • Vector RAG baseline
     └──────────────┘
                    ┌───────────────────────┐
                    │  PHASE 2:             │  • Ontology design
                    │  Knowledge Graph      │  • Entity extraction
                    │  (Weeks 4-7)          │  • Graph population
                    └───────────────────────┘  • Basic Cypher queries
                                            ┌────────────────────┐
                                            │  PHASE 3:          │  • Query router
                                            │  Hybrid Retrieval  │  • Result fusion
                                            │  (Weeks 8-10)      │  • Enhanced generation
                                            └────────────────────┘
                                                                 ┌────────────────────┐
                                                                 │  PHASE 4:          │
                                                                 │  Quality           │  • Hallucination detection
                                                                 │  (Weeks 11-13)     │  • Evaluation framework
                                                                 └────────────────────┘  • Refinement
                                                                                      ┌────────────────────┐
                                                                                      │  PHASE 5:          │
                                                                                      │  Production        │
                                                                                      │  (Weeks 14-16)     │
                                                                                      └────────────────────┘

DELIVERABLES:
  Week 3:  ✓ Working vector RAG with baseline metrics
  Week 7:  ✓ Populated knowledge graph with basic query capability
  Week 10: ✓ Integrated hybrid retrieval with query routing
  Week 13: ✓ Evaluation framework and documented performance
  Week 16: ✓ Production-ready system with API and documentation
```

---

## File Structure

```
graph-rag-starter/
├── src/
│   ├── ingestion/
│   │   ├── pdf_extractor.py      # PyMuPDF + OCR extraction
│   │   ├── chunker.py            # Entity-aware, hierarchical chunking
│   │   └── preprocessor.py       # Text cleaning, normalization
│   │
│   ├── extraction/
│   │   ├── entity_extractor.py   # GLiNER NER
│   │   ├── relation_extractor.py # LLM-based relation extraction
│   │   └── normalizer.py         # Entity resolution
│   │
│   ├── graph/
│   │   ├── neo4j_client.py       # Neo4j connection, CRUD
│   │   ├── schema.py             # Ontology enforcement
│   │   └── queries.py            # Common Cypher patterns
│   │
│   ├── retrieval/
│   │   ├── vector_retriever.py   # Qdrant search
│   │   ├── graph_retriever.py    # Neo4j traversal
│   │   ├── hybrid_retriever.py   # Result fusion
│   │   └── query_router.py       # Query classification
│   │
│   ├── generation/
│   │   ├── answer_generator.py   # LLM prompting
│   │   ├── citation_injector.py  # Source attribution
│   │   └── prompts.py            # Prompt templates
│   │
│   ├── evaluation/
│   │   ├── faithfulness.py       # Hallucination detection
│   │   ├── metrics.py            # Quality metrics
│   │   └── test_sets/            # Gold-standard Q&A pairs
│   │
│   └── api/
│       ├── main.py               # FastAPI application
│       └── routes.py             # API endpoints
│
├── config/
│   ├── ontology.yaml             # Entity/relation schema
│   ├── settings.py               # Configuration
│   └── normalization/
│       ├── regions.json          # 10 economic regions
│       └── crops.json            # Crop taxonomy
│
├── data/
│   ├── raw_pdfs/                 # Original documents
│   └── processed/                # Extracted content
│
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

*Document generated for Graph RAG implementation for Azerbaijani Agricultural Sector*
