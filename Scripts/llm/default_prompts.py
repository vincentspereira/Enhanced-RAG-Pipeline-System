"""
Default prompt templates for the RAG pipeline.
"""

class DefaultPromptTemplates:
    # QA prompt template
    QA_PROMPT = {
        "name": "qa_prompt",
        "description": "Template for question answering with context",
        "template": """Answer the question based on the provided context.

Context:
{context}

Question:
{question}

Answer:""",
        "variables": ["context", "question"]
    }
    
    # Summarization prompt template
    SUMMARIZE_PROMPT = {
        "name": "summarize_prompt",
        "description": "Template for text summarization",
        "template": """Summarize the following information in a concise manner:

{text}

Summary:""",
        "variables": ["text"]
    }
    
    # Hybrid search prompt template
    HYBRID_SEARCH_PROMPT = {
        "name": "hybrid_search_prompt",
        "description": "Template for hybrid search (semantic + keyword)",
        "template": """You are searching for information on the following query:

Query: {query}

Here are the most relevant results:
{results}

Based on these results, please provide a comprehensive answer to the query.
""",
        "variables": ["query", "results"]
    }
    
    # Query expansion prompt template
    QUERY_EXPANSION_PROMPT = {
        "name": "query_expansion_prompt",
        "description": "Template for query expansion",
        "template": """The user has asked the following question:

{query}

Generate 3-5 related search queries that might help find more comprehensive information:""",
        "variables": ["query"]
    }
    
    # Document categorization prompt template
    CATEGORIZATION_PROMPT = {
        "name": "categorization_prompt",
        "description": "Template for document categorization",
        "template": """Analyze the following text and assign appropriate categories from the available options.

Text:
{text}

Available categories:
{categories}

Assigned categories (comma-separated):""",
        "variables": ["text", "categories"]
    }
    
    # Collection
    TEMPLATES = {
        "qa_prompt": QA_PROMPT,
        "summarize_prompt": SUMMARIZE_PROMPT,
        "hybrid_search_prompt": HYBRID_SEARCH_PROMPT,
        "query_expansion_prompt": QUERY_EXPANSION_PROMPT,
        "categorization_prompt": CATEGORIZATION_PROMPT
    }
