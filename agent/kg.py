import json
import math
import re

from loguru import logger
from openai import BadRequestError
from tqdm import tqdm

from api.api_db_client import get_actions_api
from api.llm_ppl_client import calculate_perplexity
from common.common_utils import colorful, read_json
from data_loader.utils import make_prompt_text
from evaluation.utils import extract_dialog_last_valid_observation
from tool.openai_api import chatgpt

SearchNodes, SearchGraphPatterns, ExecuteSPARQL = None, None, None


def conditional_prob_q_desc(q, entity_name, desc):
    """
    Calculate P(Q | e) = P(Q, Desc(e)) / P(Desc(e))
    P(.) ∝ 1 / PPL(.) -> P(Q | e) = PPL(Desc(e)) / PPL(Q, Desc(e))
    use sigmoid function to amplify the difference.
    """
    # add a prompt for entity description.
    prompt = """
Given a entity and its description, I can make a question about it.

Entity: {entity_name}
Description: {desc}

Question: {q}
    """.strip()
    prompt_no_q = prompt.format(entity_name=entity_name, desc=desc, q="")
    prompt_with_q = prompt.format(entity_name=entity_name, desc=desc, q=q)
    res = calculate_perplexity(prompt_no_q) / calculate_perplexity(prompt_with_q)
    return 1 / (1 + math.exp(-res))


def calculate_ambiguous_score_for_entity(q, action_str: str):
    """
    Calculate ambiguity score based on entity popularity.
    Returns a score between 0 and 1, where:
        - 0 means no ambiguity (single clear match)
        - 1 means maximum ambiguity (multiple equally likely matches)
    The smaller the score, the clearer the intention
    """
    # Extract query and get search results
    query = action_str.split("SearchNodes")[1].strip()[2:-2]
    # Dict: {"entity_name": [{'mid': mid, 'score': score, 'desc': desc}, ...]}
    results = SearchNodes(query, str_mode=False)
    # add ename to each match items
    for ename in results:
        for i in range(len(results[ename])):
            results[ename][i]["entity_name"] = ename

    # 1. P(e_i)
    # Separate exact matches and partial matches
    exact_matches = []
    partial_matches = []
    for ename, matches in results.items():
        if len(matches) > 1:
            exact_matches = matches[:10]  # only one.
        elif len(matches) == 1:
            partial_matches.append(matches[0])

    # Calculate probabilities for exact matches
    exact_total_score = sum(item["score"] for item in exact_matches)
    for i in range(len(exact_matches)):
        exact_matches[i]["prior_prob"] = exact_matches[i]["score"] / exact_total_score

    # Calculate probabilities for partial matches
    partial_total_score = sum(item["score"] for item in partial_matches)
    for i in range(len(partial_matches)):
        partial_matches[i]["prior_prob"] = partial_matches[i]["score"] / partial_total_score

    # 2. P(Q | e_i) = PPL(Desc(e_j)) / PPL(Q, Desc(e_j))
    pbar = tqdm(
        total=len(exact_matches) + len(partial_matches),
        desc="Calculating conditional probabilities for entities",
        ncols=100,
        ascii=" >=",
        disable=1,
    )
    for i in range(len(exact_matches)):
        exact_matches[i]["conditional_prob"] = conditional_prob_q_desc(
            q, exact_matches[i]["entity_name"], exact_matches[i]["desc"]
        )
        pbar.update(1)
    for i in range(len(partial_matches)):
        partial_matches[i]["conditional_prob"] = conditional_prob_q_desc(
            q, partial_matches[i]["entity_name"], partial_matches[i]["desc"]
        )
        pbar.update(1)
    pbar.close()

    # 3. Calculate posterior probability
    for i in range(len(exact_matches)):
        exact_matches[i]["posterior_prob"] = (
            exact_matches[i]["prior_prob"] * exact_matches[i]["conditional_prob"]
        )
    for i in range(len(partial_matches)):
        partial_matches[i]["posterior_prob"] = (
            partial_matches[i]["prior_prob"] * partial_matches[i]["conditional_prob"]
        )

    # 4. Normalize the posterior probability using z-score and softmax
    # Calculate z-score for exact matches
    if len(exact_matches) > 0:
        exact_probs = [item["posterior_prob"] for item in exact_matches]
        exact_mean = sum(exact_probs) / len(exact_probs)
        exact_std = (sum((x - exact_mean) ** 2 for x in exact_probs) / len(exact_probs)) ** 0.5
        if exact_std == 0:
            exact_std = 1
        # Apply z-score normalization then softmax
        exact_z_scores = [(x - exact_mean) / exact_std for x in exact_probs]
        exact_exp = [math.exp(z) for z in exact_z_scores]
        exact_sum = sum(exact_exp)
        for i in range(len(exact_matches)):
            exact_matches[i]["posterior_prob"] = exact_exp[i] / exact_sum

    # Calculate z-score for partial matches
    if len(partial_matches) > 0:
        partial_probs = [item["posterior_prob"] for item in partial_matches]
        partial_mean = sum(partial_probs) / len(partial_probs)
        partial_std = (sum((x - partial_mean) ** 2 for x in partial_probs) / len(partial_probs)) ** 0.5
        if partial_std == 0:
            partial_std = 1
        # Apply z-score normalization then softmax
        partial_z_scores = [(x - partial_mean) / partial_std for x in partial_probs]
        partial_exp = [math.exp(z) for z in partial_z_scores]
        partial_sum = sum(partial_exp)
        for i in range(len(partial_matches)):
            partial_matches[i]["posterior_prob"] = partial_exp[i] / partial_sum

    # 5. Calculate entropy-based ambiguity score
    h_exact = -sum(item["posterior_prob"] * math.log2(item["posterior_prob"]) for item in exact_matches)
    h_partial = -sum(item["posterior_prob"] * math.log2(item["posterior_prob"]) for item in partial_matches)

    # Combine probabilities with different groups (e.g., 0.7 for exact, 0.3 for partial)
    EXACT_WEIGHT = 0.7
    PARTIAL_WEIGHT = 0.3

    part1 = EXACT_WEIGHT * h_exact / math.log2(len(exact_matches)) if len(exact_matches) > 1 else 0
    part2 = PARTIAL_WEIGHT * h_partial / math.log2(len(partial_matches)) if len(partial_matches) > 1 else 0
    ambiguity_score = part1 + part2

    return min(ambiguity_score, 1)


def calculate_ambiguous_score_for_intention(q, topic_entity, search_predicate_res: str):
    """
    Calculate ambiguity score based on intention.
    """
    obs = (
        search_predicate_res[1:-1]
        .replace("), (", ")@@@@(")
        .replace("(?e, ", "(")
        .replace(" ,?e)", ")")
        .split("@@@@")
    )
    if len(obs) == 0:
        return None
    if len(obs) == 1:
        return 0

    prompt = """
Given a relation and its corresponding target entity, I can make a question to ask about the target entity.

Topic Entity: Natalie Portman
Relation: (film.actor.film -> film.performance.character, "Inés")
Question: What character did Natalie Portman play in a film?

Topic Entity: {topic_entity}
Relation: {predicate}
Question: {q}
    """.strip()

    # make item list
    predicate_items = [{"predicate": item} for item in obs]
    if len(predicate_items) == 1:
        return 0

    for item in tqdm(
        predicate_items,
        desc="Calculating conditional probabilities for predicates",
        ncols=100,
        ascii=" >=",
        disable=True,
    ):
        # 1. cal ppl(q,r,e')
        item["ppl_q_r_e"] = calculate_perplexity(
            prompt.format(topic_entity=topic_entity, predicate=item["predicate"], q=q)
        )

        # 2. cal ppl(r,e')
        item["ppl_r_e"] = calculate_perplexity(
            prompt.format(topic_entity=topic_entity, predicate=item["predicate"], q="")
        )

        # 3. cal p(r_i|Q)
        ratio = item["ppl_r_e"] / item["ppl_q_r_e"]
        # ratio = ratio ** 2 - 1
        # item["p_r_given_q"] = 1 / (1 + math.exp(-ratio))
        item["ratio"] = ratio

    # normalize p(r_i|Q) using z-score
    mean_ratio = sum(item["ratio"] for item in predicate_items) / len(predicate_items)
    std_ratio = math.sqrt(
        sum((item["ratio"] - mean_ratio) ** 2 for item in predicate_items) / len(predicate_items)
    )

    # first normalize using z-score
    for item in predicate_items:
        item["z_score"] = (item["ratio"] - mean_ratio) / std_ratio

    # then convert to probability distribution using softmax
    max_z = max(item["z_score"] for item in predicate_items)
    exp_sum = sum(math.exp(item["z_score"] - max_z) for item in predicate_items)
    for item in predicate_items:
        item["posterior_prob"] = math.exp(item["z_score"] - max_z) / exp_sum
        # print(item["predicate"])
        # print(f"ppl_q_r_e: {item['ppl_q_r_e']:.4f} ppl_r_e: {item['ppl_r_e']:.4f} ratio:{item['ratio']:.4f} z_score: {item['z_score']:.4f} posterior_prob: {item['posterior_prob']:.4f}")
        # print()

    # 4. cal entropy
    entropy = -sum(item["posterior_prob"] * math.log2(item["posterior_prob"]) for item in predicate_items)

    # 5. cal ambiguity score
    ambiguity_score = entropy / math.log2(len(predicate_items))
    return min(ambiguity_score, 1)


def parse_action(text: str, db: str = None, execute=False):
    """
    Handling single quotes
        "Critics' Choice Movie Award"
        "Man of Steel's narrative location"
    """
    try:
        text = text.strip()

        action_str = text.split("Action:")
        if len(action_str) < 2:
            return "Error: No valid action found. Please provide an action after `Action:` ."

        # find the start index of SearchNodes, SearchGraphPatterns, ExecuteSPARQL, Done
        action_str = action_str[1]
        if not any(
            [i in action_str for i in ["SearchNodes", "SearchGraphPatterns", "ExecuteSPARQL", "Done"]]
        ):
            return "Error: Action no found, action must be one of [SearchNodes, SearchGraphPatterns, ExecuteSPARQL, Done], one at a time."
        _s = [action_str.find(i) for i in ["SearchNodes", "SearchGraphPatterns", "ExecuteSPARQL", "Done"]]
        _s = min([i for i in _s if i != -1])
        action_str = action_str[_s:].strip()
        if action_str.startswith("Done"):
            return "Done"

        # Handling single quotes
        if "\\'" not in action_str:
            action_str = action_str.replace("' ", "\\' ").replace("'s ", "\\'s ")
        # Handling \_
        action_str = action_str.replace("\\_", "_")

        # find the last ) as the end.
        action_str = action_str[: action_str.rfind(")") + 1]

        if execute:
            assert db is not None, "db must be provided when execute=True"
            global SearchNodes, SearchGraphPatterns, ExecuteSPARQL
            if SearchNodes is None:
                logger.info(f"init actions for db: {db}")
                SearchNodes, SearchGraphPatterns, ExecuteSPARQL = get_actions_api(db)

            # may time out, return None
            obs = eval(action_str)
            return str(obs)
        else:
            return action_str
    except Exception as e:
        # print_exc()
        err = f"Error: Action parsing error. {e.__class__.__name__}: {str(e)}"
        logger.error(f"{err}. action_str: {action_str}. Raw: {text}")
        return err


def extract_entity_from_sparql(sparql: str):
    """
    e.g. type.object.name "Alice Walker"@en -> Alice Walker
    """
    pattern = r'type.object.name "([^"]+)"@en'
    matches = re.findall(pattern, sparql)
    return matches


cwq_qtype_prediction = None


class KGAgent:
    def __init__(
        self,
        q,
        qid,
        dataset: str,
        model_name: str,
        max_round_num: int = 10,
        qtype=None,
        ambiguous_score_entity_threshold=0.6,
        ambiguous_score_intention_threshold=0.8,
        clear_q=None,
    ):
        assert dataset in [
            "webqsp",
            "cwq",
            "wqcwq",
        ], f"dataset must be one of ['webqsp', 'cwq', 'wqcwq'], but got {dataset}"
        self.q = q
        self.qid = qid
        self.db = "fb"
        self.dataset = dataset
        self.model_name = model_name
        self.ambiguous_score_entity_threshold = ambiguous_score_entity_threshold
        self.ambiguous_score_intention_threshold = ambiguous_score_intention_threshold
        self.clear_q = clear_q

        if dataset == "cwq":
            global cwq_qtype_prediction
            if cwq_qtype_prediction is None:
                cwq_qtype_prediction = read_json("agent/cwq-classification-prediction.json")
                cwq_qtype_prediction = {item["id"]: item["pred_label"] for item in cwq_qtype_prediction}
            qtype = cwq_qtype_prediction[self.qid]
        elif "-epoch" in model_name:
            qtype = None

        self.prompt_text = make_prompt_text(
            dataset, setting="kbqa" if self.clear_q is None else "kbqa-clearq", qtype=qtype
        )
        self.llm_config = {
            "max_completion_tokens": 512,
            "temperature": 0.7,
            "n": 1,
        }
        self.max_round_num = max_round_num
        self.messages = []
        self.completion_tokens = []
        self.prompt_tokens = []
        self.prediction = []
        self.ambiguous_records = []

        self._enable_entity_ambiguity = True
        self._enable_intention_ambiguity = True

    def disable_entity_ambiguity(self):
        self._enable_entity_ambiguity = False

    def disable_intention_ambiguity(self):
        self._enable_intention_ambiguity = False

    def run_cot(self):
        system_prompt = """
You are a question answering expert. Please provide your answer in JSON format. MUST contain "answer" field.
Example:
Q: what was alice walker famous for?
A: {"answer": ['Author', 'Novelist', 'Poet', 'Writer']}
""".strip()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Q: " + self.q.strip()},
        ]
        # try 3 times
        res = ["_None"]
        for _ in range(5):
            response = chatgpt(
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"},
                **self.llm_config,
            )
            try:
                res = json.loads(response["choices"][0]["message"]["content"])
                res = res["answer"]
            except:
                continue
        _d = {
            "qid": self.qid,
            "q": self.q,
            "clear_q": self.clear_q,
            "model_name": self.model_name,
            "prediction": res,
        }
        return _d

    def run(self, call_back=lambda x: input(x), plugin=True):
        """
        call_back: a function, str input, str output
        """
        if self.clear_q:
            q = self.clear_q.strip()
        else:
            q = self.q.strip()

        logger.warning(f"Running KGAgent for Question: {q}")

        messages = [
            {"role": "system", "content": self.prompt_text},
            {"role": "user", "content": "Q: " + q},
        ]

        round_id = 0
        _last_choice = ""

        # info
        completion_tokens = []
        prompt_tokens = []

        # history: all inp and out
        # history = []

        while round_id < self.max_round_num:
            if round_id > 0:
                logger.debug(f"round_id: {round_id}")

            try:
                if "-epoch" in self.model_name:
                    from common.constant import get_actor_url

                    url = get_actor_url()
                    key = "kbqa"
                else:
                    url = None
                    key = None
                response = chatgpt(
                    model=self.model_name,
                    messages=messages,
                    stop=["\nObservation", "\nThought"],
                    url=url,
                    key=key,
                    **self.llm_config,
                )
                self.model_name = response["model"]
            except BadRequestError as e:
                logger.error(f"BadRequestError: {e}")
                return
            if response is None:
                logger.error(f"response is None.")
                return
            if "usage" not in response:
                logger.error(response["error"])
                return

            prompt_tokens.append(response["usage"]["prompt_tokens"])
            completion_tokens.append(response["usage"]["completion_tokens"])

            # prepare choices
            choice = response["choices"][0]["message"]["content"].replace("\n\n", "\n").strip()

            print(colorful("LLM: ", color="yellow"), end="")
            print(choice.replace("\n", "\\n"))

            # Attempt to execute the first valid action in the actions.
            if "Action: AskForClarification" in choice:
                # Do not allow agent to guess the answer.
                # if last action is not ExecuteSPARQL, then it is not a valid action.
                if "Action: ExecuteSPARQL" in messages[-2]["content"]:
                    Observation = "Error: You can not ask for clarification after ExecuteSPARQL action."
                else:
                    _prompt_text = choice.split("Action: AskForClarification")[1].strip()[1:-1]
                    _res = call_back(_prompt_text)
                    Observation = "Clarification: " + _res.strip()
            else:
                Observation = parse_action(choice, db=self.db, execute=True)

                # time out
                if Observation == "None":
                    return

                Observation = f"Observation: {Observation}"

            # Clarification Plugin.
            if "Action: SearchNodes" in choice and plugin and self._enable_entity_ambiguity:
                ambiguous_score_entity = calculate_ambiguous_score_for_entity(q=self.q, action_str=choice)
                logger.debug(f"ambiguous score for entity: {ambiguous_score_entity}")
                self.ambiguous_records.append((round_id, "entity", ambiguous_score_entity))
                if ambiguous_score_entity > self.ambiguous_score_entity_threshold:
                    Observation += "\n(Hint: The entity may be ambiguous. Please decide whether to ask user to clarify based on the context.)"

            elif "Action: SearchGraphPatterns" in choice and plugin and self._enable_intention_ambiguity:
                entity_predict = extract_entity_from_sparql(choice)
                entity_predict = ", ".join(entity_predict)
                ambiguous_score_intention = calculate_ambiguous_score_for_intention(
                    q=self.q,
                    topic_entity=entity_predict,
                    search_predicate_res=Observation.split("Observation: ")[-1],
                )
                logger.debug(f"ambiguous score for intention: {ambiguous_score_intention}")
                self.ambiguous_records.append((round_id, "intention", ambiguous_score_intention))
                if ambiguous_score_intention > self.ambiguous_score_intention_threshold:
                    Observation += "\n(Hint: The intention may be ambiguous. Please decide whether to ask user to clarify based on the context.)"

            print(colorful("Observation or Clarification: ", color="yellow"), end="")
            print(Observation.replace("Hint: ", colorful("Hint: ", color="red")))

            if choice == _last_choice:
                messages.append({"role": "user", "content": "STOP because of repetition."})
                break
            _last_choice = choice

            if not choice.startswith("Thought: "):
                choice = "Thought: " + choice

            messages.append({"role": "assistant", "content": choice})

            if "Action: Done" not in choice:
                messages.append(
                    {
                        "role": "user",
                        "content": Observation,
                    }
                )
                round_id += 1
            else:
                # use extract_dialog_last_valid_observation to check the final answer is valid
                final_answer = extract_dialog_last_valid_observation({"dialog": messages})
                if final_answer == ["_None"]:
                    Observation = "Error: Detected Done, but the final answer is not valid. Please provide the final SPARQL query using `Action: ExecuteSPARQL('SELECT DISTINCT ?x WHERE ...)`."
                    messages.append({"role": "user", "content": Observation})
                    continue
                else:
                    messages.append({"role": "user", "content": "Stop condition detected."})
                    break

            # debug
            # break

        self.messages = messages
        self.completion_tokens = completion_tokens
        self.prompt_tokens = prompt_tokens
        self.prediction = extract_dialog_last_valid_observation({"dialog": messages})

    def to_dict(self):
        messages = [m for m in self.messages if m["role"] != "system"]
        r = {
            "qid": self.qid,
            "q": self.q,
            "model_name": self.model_name,
            "messages": messages,
            "prediction": self.prediction,
            "ambiguous_records": self.ambiguous_records,
            "ambiguous_score_entity_threshold": self.ambiguous_score_entity_threshold,
            "ambiguous_score_intention_threshold": self.ambiguous_score_intention_threshold,
        }
        if self.clear_q:
            r["clear_q"] = self.clear_q
        return r


if __name__ == "__main__":
    # python agent/kg.py

    # q="What zoo that opened earliest is there to see in Dallas, TX?"

    # q = "Who won the noble peace prize in 2007?"
    q = "Who won the noble prize"

    agent = KGAgent(
        q=q, qid="WebQTest-12_7b54b31f3e5a6273f4fd2a20e565ec6d", dataset="cwq", model_name="gpt-4o"
    )
    agent.run(call_back=lambda x: input(x + "\n> "))
    agent.run_cot()
