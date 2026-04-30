"""Conversation specialist — LUCID expert chat with no tools."""

# This specialist has no tools. It uses the CONVERSATION_SYSTEM_PROMPT
# directly with the LLM. The graph node handles invocation without
# create_react_agent since there are no tools to bind.
