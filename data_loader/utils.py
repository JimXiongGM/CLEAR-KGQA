import random
from glob import glob
from typing import Dict, List

from loguru import logger

from common.common_utils import read_json


def load_text_exclude_comment(file_p, _list=False):
    with open(file_p, "r") as f:
        lines = f.readlines()
    lines = [i for i in lines if not i.startswith("#")]
    if _list:
        return lines
    return "".join(lines).strip()


# cwq/webqsp (for FB)
TOOL_DESC_FULL_FB = load_text_exclude_comment(f"fewshot_demo/webqsp/prompt_kbqa.txt")

TOOL_DESC_SHORT_FB = """Given a question, you need to use the specific tools to interact with Freebase and write a SPARQL query to get the answer.

The following document includes the description of tools, the types of nodes and edges (KG schema), and some critical graph patterns.

1. SearchNodes(query)
Description: Search for nodes in the knowledge graph based on the surface name. 

2. ExecuteSPARQL(sparql)
Description: Execute a SPARQL query. You can explore the KG freely using this tool.

3. SearchGraphPatterns(sparql, semantic)
Description: Parameter `sparql` MUST start with "SELECT ?e WHERE". The tool will query the subgraphs with ?e as the head or tail entities, respectively, and return them together. The `semantic` parameter indicates the expected predicate semantics. If provided, the tool will sort the queried subgraphs by semantics. If not, the tool returns the entire subgraph. Note! If a predicate corresponds to multiple tail entities, this tool randomly returns one of them. It is worth noting that, in Freebase, due to the use of "Compound Value Type" (CVT) to represent an event, a one-hop relationship semantically requires two hops in Freebase. If encountering a CVT node, this tool will split the two-hop relationships involving the CVT node into predicate pairs and consider them as one-hop relationships.

Now, think and solve the following complex questions step by step."""


def load_test_data(dataset, num_each_type=None):
    assert dataset in ["webqsp", "cwq", "kqapro"], f"dataset: {dataset} not supported."
    _split = "test" if dataset != "kqapro" else "dev"
    p = f"dataset_processed/{dataset}/{_split}/*.json"
    paths = glob(p)
    data = []
    for p in paths:
        data += read_json(p)[:num_each_type]
    return data


def load_train_data(dataset):
    assert dataset in ["webqsp", "cwq", "kqapro"], f"dataset: {dataset} not supported."
    p = f"save-anno-clean/{dataset}/*/*.json"
    paths = glob(p)
    data = [read_json(p) for p in paths]
    return data


def load_fewshot_demo_dialog(dataset, qtype=None, entity=None):
    """
    base: fewshot_demo/{dataset}/dialog/*.txt
    entity: fewshot_demo/{dataset}/dialog-{entity}-entity/...
    qtype: fewshot_demo/{dataset}/dialog/{qtype}-[01/02].txt

    for cwq/kqapro:
        - `_4_shot` is used to load fixed 4-shot demo to represent all qtype.
            - kqa: QueryName / QueryRelation / QueryRelationQualifier / Verify
        - `qtype` is used to load demo by qtype.
    for webqsp, load all examples.
        - fewshot_demo/webqsp/dialog/*.txt
    """
    dir_patt = f"fewshot_demo/{dataset}/dialog/"

    if dataset == "webqsp":
        dir_patt = f"fewshot_demo/{dataset}/dialog/"
        qtype = None

    if entity:
        dir_patt = dir_patt[:-1] + f"-{entity}-entity/"

    if qtype:
        dir_patt += f"{qtype}-[0-9][0-9].txt"
    else:
        dir_patt += "*.txt"

    logger.warning(f"dir_patt: {dir_patt}")
    paths = glob(dir_patt)

    demos = []
    for p in paths:
        lines = open(p).readlines()
        lines = [i for i in lines if not i.startswith("#")]
        content = "".join(lines).strip()
        _demos = content.split("\n\n")
        demos.extend(_demos)

    if qtype:
        assert len(demos) == 2, f"if qtype is not None, len(demos) should be 2, but got {len(demos)}"

    logger.warning(f"len(demos): {len(demos)}")
    return demos


def make_prompt_text(dataset, setting="kbqa", qtype=None):
    """
    prompt_text: tool desc + demo (opentional)
    For infer directly, we need to provide full tool desc and demo (required qtype).
    For finetuned, we need to provide short tool desc.
    """

    prompt = load_text_exclude_comment(f"fewshot_demo/{dataset}/prompt_{setting}.txt", _list=True)

    # parse demos
    for i in range(len(prompt)):
        line = prompt[i].strip()
        # e.g. [fewshot_demo/cwq/{qtype}.txt] or [fewshot_demo/webqsp/chain_len1-实体+谓词歧义.txt]
        if line and line[0] == "[" and line[-1] == "]":
            _file_p = line[1:-1]
            if qtype:
                _file_p = _file_p.format(qtype=qtype)
            example = load_text_exclude_comment(_file_p)
            prompt[i] = example + "\n"

    prompt = "".join(prompt)
    return prompt.strip()


# make_prompt_text("webqsp")


def check_messages_format(messages: List[Dict]):
    assert isinstance(messages[0], dict), "messages[0] must be dict"
    assert messages[0]["role"] == "system", "messages[0] must be system"

    for i in range(1, len(messages)):
        if i % 2 == 1:
            assert messages[i]["role"] == "user", f"messages[{i}] must be user. get: {messages[i]['role']}"
        elif i % 2 == 0:
            assert (
                messages[i]["role"] == "assistant"
            ), f"messages[{i}] must be assistant. get: {messages[i]['role']}"
