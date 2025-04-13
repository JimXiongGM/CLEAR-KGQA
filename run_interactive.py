import os
import random
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

import fire
from loguru import logger
from tqdm import tqdm

from agent import DummyUser, KGAgent
from common.common_utils import read_json, save_to_json, save_to_pkl


def load_clear_q(dataset):
    p = f"save/v1/{dataset}/gpt-4o-2024-08-06/*-dummy_user.json"
    paths = glob(p)
    assert len(paths) > 0, f"No files found in {p}"
    qid_clear_q = {}
    for path in paths:
        d = read_json(path)
        qid_clear_q[d["qid"]] = d["clear_q"][0] if d["clear_q"] else None
    return qid_clear_q


def run_one(
    d,
    model_name,
    save_dir,
    plugin=True,
    cot=False,
    disable_entity_ambiguity=False,
    disable_intention_ambiguity=False,
    ambiguous_score_entity_threshold=0.6,
    ambiguous_score_intention_threshold=0.8,
):
    use_clear_q = bool(d.get("clear_q", False))
    qid = d["id"]
    dataset = d["dataset"]

    # require fields: id, sparql, entity_desc
    if "entity_desc" not in d:
        print(f"entity_desc not in d: {qid}")
        raise ValueError(f"entity_desc not in d: {qid}")

    for try_num in range(5):
        if not use_clear_q:
            dummy_user = DummyUser(
                qid=d["id"], sparql=d["sparql"], entity_desc=d["entity_desc"], model_name="gpt-4o-2024-08-06"
            )
        else:
            dummy_user = None

        kg_agent = KGAgent(
            q=d["question"],
            qid=d["id"],
            dataset=dataset,
            model_name=model_name,
            clear_q=d.get("clear_q", None),
            ambiguous_score_entity_threshold=ambiguous_score_entity_threshold,
            ambiguous_score_intention_threshold=ambiguous_score_intention_threshold,
        )
        if disable_entity_ambiguity:
            kg_agent.disable_entity_ambiguity()
        if disable_intention_ambiguity:
            kg_agent.disable_intention_ambiguity()

        if cot:
            _d = kg_agent.run_cot()
        else:
            kg_agent.run(call_back=dummy_user, plugin=plugin)
            _d = kg_agent.to_dict()

        # if invalid, continue
        if _d["prediction"] == ["_None"] and try_num != 4:
            logger.warning(f"Invalid prediction: {qid}, try {try_num}")
            continue

        # save only for interactive with the original question
        if not use_clear_q and not cot:
            save_to_json(dummy_user.to_dict(), f"{save_dir}/{qid}-dummy_user.json")
            save_to_pkl(dummy_user, f"{save_dir}/{qid}-dummy_user.pkl")

        # add answers, infer_chain for webqsp
        # add answers, compositionality_type for cwq
        _d["answers"] = d["answers"]
        _d["sparql"] = d["sparql"]
        if dataset == "cwq":
            _d["compositionality_type"] = d["compositionality_type"]
        elif dataset == "webqsp":
            _d["infer_chain"] = d["infer_chain"]

        save_to_json(_d, f"{save_dir}/{qid}-kg_agent.json")
        save_to_pkl(kg_agent, f"{save_dir}/{qid}-kg_agent.pkl")


def main(
    dataset: str = "cwq",
    model_name: str = "gpt-4o",
    note: str = "v1",
    max_workers: int = 5,
    debug=False,
    plugin=True,
    use_clear_q=False,
    cot=False,
    disable_entity_ambiguity=False,
    disable_intention_ambiguity=False,
    ambiguous_score_entity_threshold=0.6,
    ambiguous_score_intention_threshold=0.8,
):
    if dataset == "webqsp":
        paths = glob(f"./dataset_processed-v2.0/{dataset}/test-300/*.json")
    elif dataset == "cwq":
        paths = glob(f"./dataset_processed-v2.0/{dataset}/test-600/*.json")
    elif dataset == "wqcwq":
        # sample 100 items from both datasets
        paths1 = glob(f"./dataset_processed-v2.0/webqsp/test-300/*.json")
        paths2 = glob(f"./dataset_processed-v2.0/cwq/test-600/*.json")
        paths = paths1 + paths2
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    data = []
    for path in paths:
        # add "dataset" field
        _data = read_json(path)
        _ds = path.split("/")[-3]
        for i in range(len(_data)):
            _data[i]["dataset"] = _ds
        data.extend(_data)

    # only keep 100
    if disable_entity_ambiguity or disable_intention_ambiguity or dataset == "wqcwq":
        random.seed(42)
        random.shuffle(data)
        data = data[:50]

    logger.info(f"Load data: {len(data)}")

    # if debug:
    #     note = "debug"

    _model_name = model_name.split("/")[-1]

    # means run cot, no interactive, no plugin, only for clear_q.
    if cot:
        save_dir = f"save/{note}/{dataset}/{_model_name}-cot"
        plugin = False
        qid_clear_q = load_clear_q(dataset)
        for i in range(len(data)):
            qid = data[i]["id"]
            clear_q = qid_clear_q.get(qid, None)
            data[i]["clear_q"] = clear_q

    # save/{note}/{dataset}/{model_name}/{qid}.pkl and {qid}.json
    elif not plugin:
        save_dir = f"save/{note}/{dataset}/{_model_name}-no-plugin"
        if use_clear_q:
            save_dir = f"save/{note}/{dataset}/{_model_name}-no-plugin-clear-q"
            qid_clear_q = load_clear_q(dataset)
            # 注意 不是所有的qid都有clear_q
            for i in range(len(data)):
                qid = data[i]["id"]
                clear_q = qid_clear_q.get(qid, None)
                if clear_q:
                    data[i]["clear_q"] = clear_q
                else:
                    os.makedirs(f"save/{note}/{dataset}/{_model_name}-no-plugin-clear-q", exist_ok=True)
                    if os.path.exists(f"save/{note}/{dataset}/{_model_name}-no-plugin/{qid}-kg_agent.json"):
                        shutil.copy(
                            f"save/{note}/{dataset}/{_model_name}-no-plugin/{qid}-kg_agent.json",
                            f"save/{note}/{dataset}/{_model_name}-no-plugin-clear-q/{qid}-kg_agent.json",
                        )
                        data[i] = None

            data = [d for d in data if d is not None]
            logger.info(f"Loaded {len(data)} data with clear_q")
    else:
        save_dir = f"save/{note}/{dataset}/{_model_name}"
        assert use_clear_q == False, "use_clear_q is not supported for plugin mode"

    if disable_entity_ambiguity:
        save_dir += "-disable-entity"
    if disable_intention_ambiguity:
        save_dir += "-disable-intention"

    save_dir += f"-entthre{ambiguous_score_entity_threshold}-intthre{ambiguous_score_intention_threshold}"

    logger.info(f"Saving to: {save_dir}")

    skip_ids = []
    if os.path.exists(save_dir):
        paths = glob(save_dir + "/*-kg_agent.json")
        skip_ids += [p.split("/")[-1].split("-kg_agent")[0] for p in paths]

    skip_ids = set(skip_ids)
    logger.info(f"Skip id: {len(skip_ids)}")
    data = [d for d in data if d["id"] not in skip_ids]
    logger.info(f"Remain data: {len(data)}")

    if debug:
        logger.debug(f"Debug mode, only process 5 questions")
        data = data[:5]
        for d in data:
            run_one(
                d,
                model_name=model_name,
                save_dir=save_dir,
                plugin=plugin,
                cot=cot,
                disable_entity_ambiguity=disable_entity_ambiguity,
                disable_intention_ambiguity=disable_intention_ambiguity,
                ambiguous_score_entity_threshold=ambiguous_score_entity_threshold,
                ambiguous_score_intention_threshold=ambiguous_score_intention_threshold,
            )
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    run_one,
                    d,
                    model_name=model_name,
                    save_dir=save_dir,
                    plugin=plugin,
                    cot=cot,
                    disable_entity_ambiguity=disable_entity_ambiguity,
                    disable_intention_ambiguity=disable_intention_ambiguity,
                    ambiguous_score_entity_threshold=ambiguous_score_entity_threshold,
                    ambiguous_score_intention_threshold=ambiguous_score_intention_threshold,
                )
                for d in data  # [:20]
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing", ncols=100):
                try:
                    future.result()  # Get the result to propagate any exceptions
                except Exception as e:
                    print(traceback.format_exc())


def debug():
    d = {
        "id": "1",
        "question": "who won the nobel prize? (Hint: this is a ambiguous question)",
        "sparql": "SELECT DISTINCT ?x WHERE { ?e0 ns:type.object.name 'Nobel Prize in Physics'@en . ?e0 ns:award.award_category.winners ?cvt . ?cvt ns:award.award_honor.year '2007'^^xsd:gYear . ?x ns:award.award_winner.awards_won ?cvt }",
        "entity_desc": {
            "Nobel Prize in Physics": "The Nobel Prize in Physics is a yearly award given by the Royal Swedish Academy of Sciences for those ..."
        },
        "answers": ["Albert Fert", "Peter Grünberg"],
        "infer_chain": ["award.award_winner.awards_won", "award.award_category.winners"],
    }

    run_one(d, dataset="webqsp", model_name="gpt-4o", save_dir="save/debug/webqsp/gpt-4o")


if __name__ == "__main__":
    # debug()
    fire.Fire(main)
