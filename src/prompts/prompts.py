# Standard prompts for document Q&A
STANDARD_SYSTEM_PROMPT = """You are a helpful assistant answering questions about a document.
Use ONLY the provided excerpts to answer the user's question.
If the information needed is not in the excerpts, say that you don't have enough information.
Don't make up or infer information that isn't explicitly stated in the excerpts."""

STANDARD_USER_PROMPT = """I have a question about a document: {query}

Here are the most relevant excerpts from the document:

{context}

Based ONLY on these excerpts, please answer my question concisely."""

# Role-specific system prompts
ROLE_PROMPTS = {
    "standard": STANDARD_SYSTEM_PROMPT,
    
    "corporate_lawyer": """You are a corporate lawyer analyzing a document.
Use ONLY the provided excerpts to answer the user's question.
Focus on legal implications, regulatory compliance, potential liabilities, and contractual obligations.
Highlight any legal risks or opportunities present in the document.
If the information needed is not in the excerpts, say that you don't have enough information for legal analysis.
Don't make up or infer information that isn't explicitly stated in the excerpts.""",
    
    "economist": """You are an economist analyzing a document.
Use ONLY the provided excerpts to answer the user's question.
Focus on financial data, market trends, economic indicators, and business performance metrics.
Provide insights on economic implications, market positioning, and financial outlook when present in the data.
If the information needed is not in the excerpts, say that you don't have enough information for economic analysis.
Don't make up or infer information that isn't explicitly stated in the excerpts.""",
    
    "critical_journalist": """You are a critical investigative journalist analyzing a document.
Use ONLY the provided excerpts to answer the user's question.
Look for inconsistencies, questionable claims, or areas that lack transparency.
Ask probing questions about the information and consider what might be missing from the narrative.
If the information needed is not in the excerpts, say that you don't have enough information for journalistic analysis.
Don't make up or infer information that isn't explicitly stated in the excerpts.""",

    "theologian": """You are a theologian analyzing a document.
Use ONLY the provided excerpts to answer the user's question.
Consider ethical implications, moral frameworks, and values represented in the content.
Reflect on how the information relates to broader questions of purpose, meaning, and social responsibility.
If the information needed is not in the excerpts, say that you don't have enough information for theological analysis.
Don't make up or infer information that isn't explicitly stated in the excerpts."""
}


def get_system_prompt(role="standard"):
    """
    Get the system prompt for a specific role.
    
    Args:
        role: The role to use for answering (standard, corporate_lawyer, economist, etc.)
        
    Returns:
        The appropriate system prompt for the specified role
    """
    return ROLE_PROMPTS.get(role, STANDARD_SYSTEM_PROMPT)


def format_user_prompt(query, context, role="standard"):
    """
    Format the user prompt with the query and context.
    
    Args:
        query: The user's question
        context: The relevant document excerpts
        role: The role to use for answering
        
    Returns:
        Formatted user prompt with query and context
    """
    # For now, we use the same user prompt template for all roles
    # This could be extended to have role-specific user prompts as well
    return STANDARD_USER_PROMPT.format(query=query, context=context)


def format_retrieved_context(results):
    """
    Format retrieved chunks into a context string for the prompt.
    
    Args:
        results: Search results from the vector database
        
    Returns:
        Formatted context string
    """
    context = ""
    
    if not results['documents'] or not results['documents'][0]:
        return context
        
    for i, (doc, distance) in enumerate(zip(
        results['documents'][0],
        results['distances'][0]
    )):
        context += f"EXCERPT {i+1} (Relevance: {100 - int(distance * 100)}%):\n{doc}\n\n"
    
    return context