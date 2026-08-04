from pathlib import Path

root = Path('dataset')
root.mkdir(exist_ok=True)

papers = [
    ('Adaptive Retrieval for Scientific QA', 'This paper studies adaptive retrieval methods for answering scientific questions by selecting relevant passages from large corpora of technical documents.'),
    ('Dense Passage Retrieval for Long Documents', 'The study evaluates dense passage retrieval techniques that improve ranking quality for long-form academic papers.'),
    ('Hybrid Search for Biomedical Knowledge', 'A hybrid search approach combines lexical and semantic retrieval to find evidence from biomedical literature.'),
    ('Improving RAG with Query Reformulation', 'The authors propose query reformulation strategies to improve the recall of retrieval-augmented generation systems.'),
    ('Cross-Encoder Re-ranking for Research QA', 'This work demonstrates how cross-encoder re-ranking improves the top-k passages retrieved for question answering.'),
    ('Low-Latency Vector Search with FAISS', 'The paper investigates memory-efficient indexing methods for approximate nearest neighbor search in large document collections.'),
    ('ChromaDB for Semantic Document Retrieval', 'The authors benchmark ChromaDB as a vector store for semantic retrieval over academic text corpora.'),
    ('Transformer Embeddings for Scientific Text', 'This study compares transformer-based embeddings for retrieving scientific papers and technical chapters.'),
    ('Sentence-BERT for Research Paper Search', 'Sentence-BERT is used to retrieve relevant paragraphs for answering questions about scholarly articles.'),
    ('Multi-Vector Retrieval for Knowledge Bases', 'The paper proposes multi-vector encodings to represent nuanced technical concepts in literature search.'),
    ('Graph-Based Retrieval for Citation Networks', 'The authors link document embeddings with citation graphs to improve retrieval over related papers.'),
    ('Contrastive Learning for Domain Retrieval', 'Contrastive objectives are used to train retrieval encoders tailored to academic and scientific domains.'),
    ('LLM Grounding with Retrieved Evidence', 'This work studies how language models can be grounded by retrieved evidence during scientific question answering.'),
    ('Chunking Strategies for Long Contexts', 'The paper evaluates chunking and overlap strategies for processing long research articles in retrieval systems.'),
    ('Fine-Tuning Embeddings for Legal Documents', 'The authors fine-tune embedding models for legal and policy retrieval tasks with strong domain specificity.'),
    ('Question Answering over Technical Manuals', 'This study explores question answering over manuals by combining retrieval with structured summaries.'),
    ('Evaluation Metrics for RAG Systems', 'The paper proposes metrics to evaluate faithfulness, relevance, and groundedness in RAG pipelines.'),
    ('Memory-Efficient Inference in Vector Search', 'The study focuses on reducing memory overhead when building and querying large-scale vector stores.'),
    ('Summarization-Augmented Retrieval', 'The authors show that generated summaries help retrieve supporting evidence for complex questions.'),
    ('Robust Retrieval under Query Ambiguity', 'This work examines how retrieval systems behave when user questions are vague or ambiguous.'),
    ('Domain Adaptation for Scientific Embeddings', 'The paper discusses domain adaptation of embedding models for chemistry, biology, and computer science.'),
    ('Open-Domain QA with Retrieved Passages', 'The authors build an open-domain QA system that answers using retrieved passages rather than direct memorization.'),
    ('Neural Ranking for Academic Search', 'This paper presents neural ranking methods for retrieving relevant papers from a large literature database.'),
    ('Constrained Generation for Evidence-Based Answers', 'The authors propose constrained generation to ensure answers remain grounded in cited evidence.'),
    ('Multi-Hop Retrieval for Complex Questions', 'The study focuses on retrieving multiple evidence chunks to answer questions that require multi-step reasoning.'),
    ('Knowledge Distillation for Retrieval Models', 'The paper proposes distilling large retrieval models into smaller models for low-cost deployment.'),
    ('Retrieval over Heterogeneous Documents', 'This work combines text, tables, and figures into a unified retrieval pipeline for technical reports.'),
    ('Persistent Memory for RAG Agents', 'The authors propose persistent memory modules that help RAG agents retrieve prior evidence across sessions.'),
    ('Adaptive Chunking for Scientific Articles', 'The paper studies how chunk sizes should vary with section type and document structure.'),
    ('Dense Retrieval for Code Documentation', 'The study evaluates dense retrieval methods over software documentation and README files.'),
    ('Trustworthy QA with Source Attribution', 'This work focuses on attributing answers to specific passages for more trustworthy question answering.'),
    ('Incremental Indexing for Research Collections', 'The authors present methods for maintaining vector indexes as new papers are added to a corpus.'),
    ('Query Expansion for Academic Search', 'The paper evaluates query expansion methods that improve recall for technical search tasks.'),
    ('Semantic Similarity for Paper Recommendation', 'This study uses semantic similarity to recommend related research papers to readers.'),
    ('Evidence Selection in Multi-Document QA', 'The method selects the most useful evidence across many documents before answering a question.'),
    ('Compression of Embedding Indexes', 'The paper studies compression strategies that preserve retrieval quality while reducing index size.'),
    ('Noise Robustness in Retrieval Pipelines', 'This work examines how retrieval pipelines behave when documents contain noisy or incomplete text.'),
    ('Combining BM25 and Dense Retrieval', 'The authors evaluate combining traditional lexical retrieval with modern dense retrieval.'),
    ('Contextualized Embeddings for Citation Search', 'The study uses contextualized embeddings to retrieve papers relevant to a given citation context.'),
    ('Few-Shot Retrieval for Scientific Domains', 'The paper applies few-shot adaptation strategies to improve retrieval in specialized domains.'),
    ('Relevance Feedback in Research Search', 'The authors incorporate user feedback into iterative retrieval to refine search results.'),
    ('Evaluation of Retrieval-Augmented LLMs', 'This work benchmarks the impact of retrieval on factuality and answer quality in large language models.'),
    ('Instruction-Tuned Retrievers for QA', 'The paper trains instruction-tuned retrievers that better align retrieval results with user questions.'),
    ('Temporal Retrieval for Emerging Topics', 'The study focuses on retrieving recent papers and updates related to fast-moving research areas.'),
    ('Scientific Claim Verification with Retrieval', 'This work uses retrieval to verify claims made in scientific conversations and summaries.')
]

for index, (title, abstract) in enumerate(papers, start=1):
    paper_path = root / f'paper_{index:02d}.txt'
    paper_path.write_text(f'Title: {title}\nAbstract: {abstract}\nKeywords: retrieval, embeddings, research, question answering\n', encoding='utf-8')

print(f'Created {len(papers)} papers in {root}')
