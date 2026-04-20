"""Canonical n8n node type constants.

Single source of truth for AI/LLM node type identifiers used across
quality scoring, ingestion, and detection modules.
"""

# Node types that represent AI/LLM agents (nodes that have prompts)
AI_NODE_TYPES = {
    "@n8n/n8n-nodes-langchain.agent",
    "@n8n/n8n-nodes-langchain.chainLlm",
    "@n8n/n8n-nodes-langchain.chainSummarization",
    "@n8n/n8n-nodes-langchain.chainRetrievalQa",
    "n8n-nodes-base.openAi",
    "n8n-nodes-base.anthropic",
}

# LM nodes are model configuration only — they connect to agents
# but don't need system prompts, so excluded from agent scoring
LM_CONFIG_NODE_TYPES = {
    "@n8n/n8n-nodes-langchain.lmChatOpenAi",
    "@n8n/n8n-nodes-langchain.lmChatAnthropic",
    "@n8n/n8n-nodes-langchain.lmChatGoogleGemini",
    "@n8n/n8n-nodes-langchain.lmChatAzureOpenAi",
    "@n8n/n8n-nodes-langchain.lmChatOllama",
    "@n8n/n8n-nodes-langchain.lmChatMistral",
    "@n8n/n8n-nodes-langchain.lmChatGroq",
}

# Union of AI agent + LM config types (for broad AI node detection)
ALL_AI_NODE_TYPES = AI_NODE_TYPES | LM_CONFIG_NODE_TYPES

# Substring patterns for fuzzy AI node identification
AI_TYPE_KEYWORDS = {"langchain", "openai", "anthropic", "llm", "chat"}
