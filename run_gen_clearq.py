import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

import fire
from loguru import logger
from tqdm import tqdm

from agent import DummyUser
from common.common_utils import read_json, save_to_json


def run_one(path):
    user = DummyUser.load_from_file(path)
    clear_q = user.gen_clear_q()
    new_dict = user.to_dict()
    new_dict["clear_q"] = clear_q
    save_to_json(new_dict, path, _print=True)


def main(
    pattern: str = "save/debug/webqsp/gpt-4o-2024-11-20/*-dummy_user.json",
    max_workers: int = 1,
    debug=False,
):

    paths = glob(pattern)
    assert len(paths) > 0, f"No files found in {pattern}"
    logger.info(f"Found {len(paths)} files")

    data = []
    for path in paths:
        d = read_json(path)
        if "clear_q" in d:
            continue
        data.append(d)
    logger.info(f"Loaded {len(data)} data. Skip {len(paths) - len(data)} data")

    if debug:
        logger.debug(f"Debug mode, only process 5 questions")
        for p in paths[:5]:
            run_one(p)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_one, p) for p in paths]  # [:20]

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing questions", ncols=100
            ):
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
