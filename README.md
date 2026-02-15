# Multi-Agent Grounded RAG System

A LangGraph-based multi-agent system that generates strictly grounded
answers from local documents using FAISS, with citation enforcement,
retry logic, observability, and evaluation support.

The system does **not hallucinate**.\
If the answer is not supported by retrieved sources, it returns:

Not found in the sources.

------------------------------------------------------------------------

## Architecture

Agents:

-   1. Guardrails Agent -- blocks harmful or override attempts
-   2. Planner Agent -- interprets user intent and defines workflow
-   3. Retriever Agent -- retrieves evidence from FAISS
-   4. Writer Agent -- synthesizes structured answers from notes
-   5. Verifier Agent -- enforces grounding and citation validation
-   6. Query Rewriter Agent -- retry mechanism for improved retrieval

------------------------------------------------------------------------

## Requirements

-   Python 3.10
-   Virtual environment recommended

Install dependencies:

pip install -r requirements.txt

------------------------------------------------------------------------

## Run the Application

streamlit run app/streamlit_app.py

------------------------------------------------------------------------

## Logs

All runs are stored in:

logs/runs.jsonl

Each record includes:

-   task
-   final answer
-   trace
-   notes
-   latency_ms
-   timestamp_utc

------------------------------------------------------------------------

## Evaluation

Evaluation cases:

eval/questions.json

Run evaluation:

python eval/run_eval.py

Results saved to:

eval/results.json

------------------------------------------------------------------------

## System Rules

-   Answers must be grounded in retrieved sources
-   Citations are required
-   No general knowledge fallback
-   No hallucinations
-   If unsupported → Not found in the sources.
