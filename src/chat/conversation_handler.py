import datetime
from typing import List, Dict, Any, Optional

class ConversationHandler:
    def __init__(self, max_history=5):
        self.history = []
        self.max_history = max_history
        
    def add_exchange(self, user_query, assistant_response, context_used=None):
        """
        Add a conversation exchange to the history.
        
        Args:
            user_query: The user's question
            assistant_response: The assistant's response
            context_used: Optional context used to generate the response
        """
        self.history.append({
            "user_query": user_query,
            "assistant_response": assistant_response,
            "context_used": context_used,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Trim history if needed
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
            
    def get_conversation_context(self):
        """
        Get the conversation history formatted for context.
        
        Returns:
            Formatted conversation history string
        """
        if not self.history:
            return ""
            
        context = "Previous conversation:\n\n"
        for i, exchange in enumerate(self.history):
            context += f"User: {exchange['user_query']}\n"
            context += f"Assistant: {exchange['assistant_response']}\n\n"
            
        return context
    
    def format_conversational_prompt(self, current_query, retrieved_context):
        """
        Format a prompt that includes conversation history.
        
        Args:
            current_query: The current user query
            retrieved_context: The retrieved document context
            
        Returns:
            Prompt with conversation history
        """
        conversation_context = self.get_conversation_context()
        
        prompt = f"""I have a question about a document: {current_query}

Here are the most relevant excerpts from the document:

{retrieved_context}

"""
        
        if conversation_context:
            prompt += f"""
For reference, here is our previous conversation:
{conversation_context}
"""
            
        prompt += """
Based on the document excerpts and our conversation history (if relevant), please answer my current question."""
        
        return prompt
        
    def detect_follow_up_question(self, query):
        """
        Detect if the current query is a follow-up to the previous conversation.
        
        Args:
            query: The current query
            
        Returns:
            True if this appears to be a follow-up question
        """
        if not self.history:
            return False
            
        # Check for signs of a follow-up question
        follow_up_indicators = [
            "what about", "how about", "and", "also", "what else", 
            "can you", "tell me more", "additionally", "furthermore",
            "why", "how come", "when did", "where is", "who is",

            # Nederlandse varianten
            "wat dacht je van", "hoe zit het met", "en", "ook", 
            "wat nog meer", "kun je", "vertel me meer", "daarnaast", 
            "bovendien", "waarom", "hoe komt het dat", "wanneer heeft", 
            "waar is", "wie is"
        ]
        
        # Check for pronouns that might refer to previous context
        pronouns = ["it", "they", "them", "this", "that", "these", "those", "he", "she",
                    
                    # Nederlandse varianten
                    "het", "zij", "hen", "dit", "dat", "deze", "die", "hij",
                    ]
        
        query_lower = query.lower()
        
        # Check if the query starts with a follow-up indicator
        for indicator in follow_up_indicators:
            if query_lower.startswith(indicator):
                return True
                
        # Check if the query contains pronouns
        for pronoun in pronouns:
            pronoun_with_boundaries = f" {pronoun} "  # Add spaces to match whole words
            if f" {pronoun} " in f" {query_lower} ":
                return True
                
        # If the query is very short, it's likely a follow-up
        if len(query.split()) <= 3:
            return True
            
        return False