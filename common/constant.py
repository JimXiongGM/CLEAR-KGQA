import os

from loguru import logger

ACTOR_URLS = [
    "http://localhost:28000/v1",
    "http://localhost:28001/v1",
]
_actor_index = 0


def get_actor_url():
    global _actor_index
    url = ACTOR_URLS[_actor_index]
    _actor_index = (_actor_index + 1) % len(ACTOR_URLS)
    return url


# If the tool is deployed on a remote machine.
if "KBQA_API_IP" not in os.environ:
    logger.warning("KBQA_API_IP not set. Using 127.0.0.1 as default.")
    os.environ["KBQA_API_IP"] = "127.0.0.1"
else:
    logger.warning(f"KBQA_API_IP: {os.environ['KBQA_API_IP']}")

api_ip = os.environ["KBQA_API_IP"]
API_SERVER_FB = f"http://{api_ip}:19901"

DEFAULT_PPL_SERVER = "http://localhost:26000"

"""
Load tool description for different DBs. We use short description for the fine-tuned LLMs.
"""

TOOL_DESC_SHORT_FB = """Given a question, you need to use the specific tools to interact with Freebase and write a SPARQL query to get the answer.

The following document includes the description of tools, the types of nodes and edges (KG schema), and some critical graph patterns.

1. SearchNodes(query)
Description: Search for nodes in the knowledge graph based on the surface name. 

2. ExecuteSPARQL(sparql)
Description: Execute a SPARQL query. You can explore the KG freely using this tool.

3. SearchGraphPatterns(sparql, semantic)
Description: Parameter `sparql` MUST start with "SELECT ?e WHERE". The tool will query the subgraphs with ?e as the head or tail entities, respectively, and return them together. The `semantic` parameter indicates the expected predicate semantics. If provided, the tool will sort the queried subgraphs by semantics. If not, the tool returns the entire subgraph. Note! If a predicate corresponds to multiple tail entities, this tool randomly returns one of them. It is worth noting that, in Freebase, due to the use of "Compound Value Type" (CVT) to represent an event, a one-hop relationship semantically requires two hops in Freebase. If encountering a CVT node, this tool will split the two-hop relationships involving the CVT node into predicate pairs and consider them as one-hop relationships.

Now, think and solve the following complex questions step by step."""
