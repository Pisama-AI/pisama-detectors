"""Shared n8n workflow node analysis utilities.

Centralizes node identification, agent discovery, and connection graph
building that was previously duplicated across 5+ modules.
"""

from typing import Any, Dict, List

from pisama_detectors.detection._n8n_constants import (
    AI_NODE_TYPES,
    AI_TYPE_KEYWORDS,
    ALL_AI_NODE_TYPES,
    LM_CONFIG_NODE_TYPES,
)


def is_ai_node_type(node_type: str) -> bool:
    """Return True if *node_type* represents an AI / LLM node.

    Checks both the canonical set and substring keyword matches.
    """
    if node_type in ALL_AI_NODE_TYPES:
        return True
    lower = node_type.lower()
    return any(kw in lower for kw in AI_TYPE_KEYWORDS)


def is_agent_node(node: Dict[str, Any]) -> bool:
    """Check if a node is an AI/agent node that should be scored.

    Returns True for agent nodes that have prompts.
    Returns False for LM config nodes (model configuration only).
    """
    node_type = node.get("type", "")
    if node_type in LM_CONFIG_NODE_TYPES:
        return False
    return node_type in AI_NODE_TYPES


def has_ai_language_model_input(node: Dict[str, Any], workflow: Dict[str, Any]) -> bool:
    """Check if a node has incoming ai_languageModel connections (sub-node LLM provider)."""
    node_name = node.get("name", "")
    connections = workflow.get("connections", {})
    for _src_name, conn_data in connections.items():
        if not isinstance(conn_data, dict):
            continue
        for output_group in conn_data.get("ai_languageModel", []):
            if isinstance(output_group, list):
                for conn in output_group:
                    if isinstance(conn, dict) and conn.get("node") == node_name:
                        return True
    return False


def find_agent_nodes(workflow: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Find agent nodes by type AND by incoming ai_languageModel connections."""
    nodes = workflow.get("nodes", [])
    agent_nodes = []
    for n in nodes:
        node_type = n.get("type", "")
        # Skip LM config nodes — they are model settings, not agents
        if node_type in LM_CONFIG_NODE_TYPES:
            continue
        if node_type in AI_NODE_TYPES:
            agent_nodes.append(n)
        elif has_ai_language_model_input(n, workflow):
            agent_nodes.append(n)
    return agent_nodes


def build_connection_map(
    workflow_json: Dict[str, Any],
) -> Dict[str, List[str]]:
    """Map source-node name -> list of target-node names."""
    connections: Dict[str, List[str]] = {}
    raw = workflow_json.get("connections", {})
    for source_name, outputs in raw.items():
        targets: List[str] = []
        for _output_type, output_groups in outputs.items():
            for group in output_groups:
                if isinstance(group, list):
                    for link in group:
                        if isinstance(link, dict) and "node" in link:
                            targets.append(link["node"])
        connections[source_name] = targets
    return connections


def count_ai_agents(workflow: Dict[str, Any]) -> int:
    """Count AI-related nodes in a workflow (agents + LM config nodes).

    Includes both agent nodes and LM configuration nodes, since in n8n
    each agent typically has a paired LM config node.
    """
    return sum(1 for n in workflow.get("nodes", []) if is_ai_node_type(n.get("type", "")))
