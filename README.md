# RAG Market Intelligence for Equity Brokers

A retrieval-augmented generation (RAG) proof of concept that lets brokers ask plain-English questions over the morning's **financial news, SEC 10-K/10-Q filings and price data**, and get answers that **cite their sources**. Built and evaluated end to end: baseline pipeline, prompt optimization, configuration tuning and a final inference pipeline.

Project 3 of the MIT program *AI & Data Science: Leveraging Responsible AI, Data and Statistics for Practical Impact*, delivered through Great Learning. The business case is a fictional brokerage, "GlobalEdge".

## The problem

A broker can read 15–20 of the 60–80 articles and filings published before the first client call, about 20–30% of the morning's content. Recommendations get made without today's signals, and when a client pushes back the broker can't point to a source.

## What I built

- **Ingestion:** 733 unique news articles, 27 SEC filings split into 7,590 passages, and six months of prices for 23 instruments (stocks, FX, commodities, crypto). Loaded with LangChain `CSVLoader` / `TextLoader` and split with `RecursiveCharacterTextSplitter`.
- **Retrieval:** OpenAI `text-embedding-3-small` embeddings in a local **Chroma** vector store, with contextual chunk headers, entity-aware and recency-aware retrieval, and price rows reshaped into ticker-month documents.
- **Generation:** `gpt-4o-mini` with a prompt that forces every claim to cite a source tag (`[S1]`, `[S2]`…), quote figures with their period, and say explicitly when the context doesn't cover part of a question.
- **Evaluation:** **DeepEval** with `gpt-4o` as the judge (G-Eval metrics: relevance, faithfulness, completeness, answer relevancy, source attribution), on 6 held-out questions with reference answers plus 5 open-ended broker questions.
- **Prompt optimization:** DeepEval's **GEPA** optimizer.
- **Configuration tuning:** three configurations varying chunk size, overlap, top-k and generation parameters.

## Results

| Approach | Golden set | Broker cases | Composite | Avg latency |
|---|---|---|---|---|
| Baseline prompt (reference) | 0.888 | 0.916 | 0.902 | 3.5 s |
| **Config 2 – Balanced** (selected) | 0.894 | 0.901 | **0.897** | 4.0 s |
| Config 1 – Precision | 0.895 | 0.884 | 0.889 | 3.6 s |
| GEPA-optimized prompt | 0.876 | 0.900 | 0.888 | 3.5 s |
| Config 3 – Context-rich | 0.890 | 0.829 | 0.859 | 3.9 s |

Selected configuration: 1,200-character chunks, 200 overlap, k = 6, temperature 0.1, top-p 0.9, max 900 tokens. The project brief required the production pipeline to use the optimized prompt, so the baseline row stays as a reference. Differences under 0.01 are within the judge's run-to-run noise, and ties were broken by faithfulness on broker questions. Full scores per question are in [`artifacts/`](artifacts/).

## What I learned

- **Data engineering moved quality more than model tuning.** Chunk headers, entity- and recency-aware retrieval and the ticker-month price documents fixed most wrong-period answers. Prompt optimization and generation parameters moved scores by less than 0.03.
- **Financial figures are the riskiest answers.** Filings show several periods side by side in flattened tables, and four of the five approaches pulled a figure from the wrong period at least once. The LLM judge scored one of those answers 1.00 on faithfulness, so LLM-based evaluation alone isn't enough for numbers a client might act on.
- **More context wasn't safer.** The context-rich configuration matched reference answers slightly better but was the least faithful on broker questions (0.76).
- **Automated prompt optimization can drop compliance language.** GEPA improved its own validation score by 0.004 but removed the "not personalized investment advice" statement, because no metric rewarded it. The metric has to encode the policy.

## Run it

1. `pip install -r requirements.txt` (Python 3.12)
2. Copy `config.example.json` to `config.json` and add your OpenAI API key.
3. Place the course datasets in `AI&DS_Project_3_Datasets/`. They belong to the program and aren't included here.
4. Run `rag_market_intelligence.ipynb` top to bottom. It also runs on Google Colab (see the setup cell).

## Stack

Python · LangChain · Chroma · OpenAI (`text-embedding-3-small`, `gpt-4o-mini`, `gpt-4o` as judge) · DeepEval (G-Eval, GEPA) · pandas · scikit-learn · Matplotlib
