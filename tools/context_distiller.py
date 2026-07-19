"""
tools/context_distiller.py
Advanced high-signal text compression engine for context optimization.
"""
import re

def distill_dynamic_context(user_query: str, raw_memory: str, project_brief: str, max_tokens: int = 800) -> str:
    """
    Ranks and strips low-signal text tokens out of raw memory blocks
    to protect the local LLM from context dilution and token bloat.
    """
    if not raw_memory:
        return ""
        
    # Extract clean target tokens from user query
    query_words = set(re.findall(r'\w+', user_query.lower()))
    
    # Split raw memory blocks into discrete interaction events
    blocks = raw_memory.split("---")
    scored_blocks = []
    
    for block in blocks:
        if not block.strip():
            continue
        # Calculate overlapping keyword density matching the user's intent
        block_words = re.findall(r'\w+', block.lower())
        matches = sum(1 for word in block_words if word in query_words)
        
        # Calculate dynamic ranking density score
        score = matches / max(len(block_words), 1)
        scored_blocks.append((score, block.strip()))
        
    # Sort context blocks based on true semantic signal density
    scored_blocks.sort(key=lambda x: x[0], reverse=True)
    
    # Pack high-scoring blocks tightly within target token envelope
    condensed_text = []
    current_length = 0
    
    for score, text in scored_blocks:
        estimated_tokens = len(text.split())
        if current_length + estimated_tokens > max_tokens:
            break
        condensed_text.append(text)
        current_length += estimated_tokens
        
    if not condensed_text:
        return ""
        
    return "\n\n[DISTILLED HIGH-SIGNAL INTERACTIONS]:\n" + "\n---\n".join(condensed_text)