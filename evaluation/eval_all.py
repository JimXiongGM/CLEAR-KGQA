import argparse
import string
from copy import deepcopy
from datetime import date
from glob import glob
from typing import List

import regex

from common.common_utils import colorful, read_json, underline_text
from evaluation.utils import clean_xsd, extract_dialog_last_valid_observation
from evaluation.webqsp import CalculatePRF1, cal_metrics


def cal_dialog_info(data):
    """
    return:
        ave_turn: average all dialog turns
        success_rate: end with "Stop condition detected."
        ave_turn_done: average dialog turns for success dialog
    """
    try:
        # exclude the first "You are an AI assistant."
        ave_turn = sum([len(d["dialog"]) - 1 for d in data]) / len(data)
        success_rate = sum([d["dialog"][-1]["content"] == "Stop condition detected." for d in data]) / len(
            data
        )
        num_success = sum([d["dialog"][-1]["content"] == "Stop condition detected." for d in data])
        ave_turn_done = (
            sum(
                [
                    len(d["dialog"]) - 1
                    for d in data
                    if d["dialog"][-1]["content"] == "Stop condition detected."
                ]
            )
            / num_success
        )
        return round(ave_turn, 2), round(success_rate, 2), round(ave_turn_done, 2)
    except:
        return 0, 0, 0


def evaluation_webqsp(save_dir: str):
    """
    metrics: precision, recall, average_f1, f1_average, accuracy, hit1, hit5, hit10
    """
    print(f"save_dir: {underline_text(save_dir)}")
    paths = glob(save_dir + "/*-kg_agent.json")
    if not paths:
        raise FileNotFoundError(f"no file in {save_dir}")
    data = []
    for p in paths:
        d = read_json(p)
        d["prediction"] = (
            extract_dialog_last_valid_observation(d) if "prediction" not in d else d["prediction"]
        )

        data.append(d)

    # [("type", data), ...]
    _data1 = [d for d in data if len(d["infer_chain"]) == 1]
    _data2 = [d for d in data if len(d["infer_chain"]) == 2]
    tp_data = [
        ("chain_len_1", _data1),
        ("chain_len_2", _data2),
        ("total      ", deepcopy(data)),
    ]

    # print len(data) for each type
    for tp, data in tp_data:
        print(f"{tp}: {len(data)}", end="\t")
    print()

    all_ave_f1s = []

    for idx, (tp, data) in enumerate(tp_data):
        golden_answers = [clean_xsd(i["answers"]) for i in data]
        predictions = [clean_xsd(i["prediction"]) for i in data]

        # cal metrics for webqsp and cwq; cal acc for kqapro and metaqa
        (
            precision,
            recall,
            average_f1,
            f1_average,
            accuracy,
            hit1,
            hit5,
            hit10,
            average_random_hit,
        ) = cal_metrics(golden_answers, predictions)
        if idx == 0:
            head = "type\tprecision\trecall\tave_f1\tf1_ave\taccuracy\thit@1\thit@5\thit@10\tRandomHit"
            if data and "dialog" in data[0]:
                head = head + "\tave_turn\tsuccess_rate\tave_turn_done"
            print(colorful(head, "red"))

        line = f"{tp}\t{precision}\t{recall}\t{average_f1}\t{f1_average}\t{accuracy}\t{hit1}\t{hit5}\t{hit10}\t{average_random_hit}"

        # turn info
        if data and "dialog" in data[0]:
            ave_turn, success_rate, ave_turn_done = cal_dialog_info(data)
            line = line + f"\t{ave_turn}\t{success_rate}\t{ave_turn_done}"

        print(line)
        all_ave_f1s.append(average_f1)
    print()
    line = "\t".join([str(i) for i in all_ave_f1s]) + "\t" + str(average_f1)
    print(line)


# Normalization from SQuAD evaluation script https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def ems(prediction, ground_truths):
    """
    copy from DeCAF:
    """
    return max([exact_match_score(prediction, gt) for gt in ground_truths])


def cal_exact_match_score(golden_answers: List[str], predictions: List[str]):
    if len(golden_answers) == 0:
        if len(predictions) == 0:
            return 1
        return 0
    for prediction in predictions:
        assert isinstance(prediction, str)
        em_score = ems(prediction, golden_answers)
        if em_score:
            return em_score
    return 0


def evaluation_cwq(save_dir: str):
    """
    metrics: exact_match
    e.g.
        save/v1/cwq/gpt-4o-2024-11-20/WebQTest-1251_d08336daa754523cf13934359b794632.json
        save_dir should be save/v1/cwq/gpt-4o-2024-11-20
    """
    print(f"save_dir: {underline_text(save_dir)}")
    paths = glob(save_dir + "/*-kg_agent.json")
    if not paths:
        raise FileNotFoundError(f"no file in {save_dir}")
    data = []
    for p in paths:
        d = read_json(p)
        d["prediction"] = (
            extract_dialog_last_valid_observation(d) if "prediction" not in d else d["prediction"]
        )
        data.append(d)

    # [("type", data), ...]
    _data1 = [d for d in data if d["compositionality_type"] == "conjunction"]
    _data2 = [d for d in data if d["compositionality_type"] == "composition"]
    _data3 = [d for d in data if d["compositionality_type"] == "comparative"]
    _data4 = [d for d in data if d["compositionality_type"] == "superlative"]
    tp_data = [
        ("conjunction", _data1),
        ("composition", _data2),
        ("comparative", _data3),
        ("superlative", _data4),
        ("total      ", deepcopy(data)),
    ]

    # print len(data) for each type
    for tp, data in tp_data:
        print(f"{tp}: {len(data)}", end="\t")
    print()

    all_ave_f1s = []

    for idx, (tp, data) in enumerate(tp_data):
        golden_answers = [clean_xsd(i["answers"]) for i in data]
        predictions = [clean_xsd(i["prediction"]) for i in data]

        (
            precision,
            recall,
            average_f1,
            f1_average,
            accuracy,
            hit1,
            hit5,
            hit10,
            average_random_hit,
        ) = cal_metrics(golden_answers, predictions)
        all_ave_f1s.append(average_f1)

        # metrics: exact match, As long as any one of the golden answers is in the predicted answers,
        # it will be considered correct.
        # cwq paper code adds all aliases to golden answers to calculate "exact match"
        golden_answers_add_alias = []
        for d in data:
            aliases = d["answers"] if isinstance(d["answers"], list) else [d["answers"]]
            aliases = sorted(set(aliases))
            golden_answers_add_alias.append(aliases)

        assert len(golden_answers_add_alias) == len(predictions)

        exact_match = (
            sum([cal_exact_match_score(gs, ps) for gs, ps in zip(golden_answers_add_alias, predictions)])
            / len(predictions)
            if len(predictions) > 0
            else 0
        )
        exact_match = round(exact_match * 100, 2)

        # print q types as header, acc as value
        if idx == 0:
            head = "type\tprecision\trecall\tave_f1\tf1_ave\taccuracy\thit@1\thit@5\thit@10\texact_match"
            if data and "dialog" in data[0]:
                head = head + "\tave_turn\tsuccess_rate\tave_turn_done"
            print(colorful(head, "red"))

        line = f"{tp}\t{precision}\t{recall}\t{average_f1}\t{f1_average}\t{accuracy}\t{hit1}\t{hit5}\t{hit10}\t{exact_match}"

        print(line)
    print()
    line = "\t".join([str(i) for i in all_ave_f1s]) + "\t" + str(exact_match)
    print(line)


# --- kqa


def whether_equal(answer: str, pred: str):
    """
    check whether the two arguments are equal as attribute value
    """
    if isinstance(answer, list):
        answer = answer[0] if answer else "_None"
    if isinstance(pred, list):
        pred = pred[0] if pred else "_None"

    pred = pred.replace("_", " ")
    answer = answer.replace("_", " ")

    def truncate_float(x):
        # convert answer from '100.0 meters' to '100 meters'
        try:
            v, *u = x.split()
            v = float(v)
            if v - int(v) < 1e-5:
                v = int(v)
            if len(u) == 0:
                x = str(v)
            else:
                x = "{} {}".format(str(v), " ".join(u))
        except:
            pass
        return x

    def equal_as_date(x, y):
        # check whether x and y are equal as type of date or year
        try:
            x_split = x.split("-")
            y_split = y.split("-")
            if len(x_split) == 3:
                x = date(int(x_split[0]), int(x_split[1]), int(x_split[2]))
            else:
                x = int(x)
            if len(y_split) == 3:
                y = date(int(y_split[0]), int(y_split[1]), int(y_split[2]))
            else:
                y = int(y)
            if isinstance(x, date) and isinstance(y, date):
                return x == y
            else:
                x = x.year if isinstance(x, date) else x
                y = y.year if isinstance(y, date) else y
                return x == y
        except:
            return False

    answer = truncate_float(answer)
    pred = truncate_float(pred)
    if equal_as_date(answer, pred):
        return True
    else:
        return answer == pred


def evaluation_wqcwq(save_dir: str):
    """
    metrics: precision, recall, average_f1, f1_average, accuracy, hit1, hit5, hit10
    """
    paths = glob(save_dir + "/*-kg_agent.json")
    if not paths:
        raise FileNotFoundError(f"no file in {save_dir}")
    data = []
    for p in paths:
        d = read_json(p)
        d["prediction"] = (
            extract_dialog_last_valid_observation(d) if "prediction" not in d else d["prediction"]
        )
        d["dialog"] = d["messages"]
        data.append(d)

    print(f"save_dir: {underline_text(save_dir)}  Count: {len(data)}")

    golden_answers = [clean_xsd(i["answers"]) for i in data]
    predictions = [clean_xsd(i["prediction"]) for i in data]

    head = "precision\trecall\tave_f1\tf1_ave\taccuracy\thit@1\thit@5\thit@10\tRandomHit"
    head = head + "\tave_turn\tsuccess_rate\tave_turn_done"
    # print(colorful(head, "red"))

    # cal metrics for webqsp and cwq; cal acc for kqapro and metaqa
    (
        precision,
        recall,
        average_f1,
        f1_average,
        accuracy,
        hit1,
        hit5,
        hit10,
        average_random_hit,
    ) = cal_metrics(golden_answers, predictions)

    line = f"{precision}\t{recall}\t{average_f1}\t{f1_average}\t{accuracy}\t{hit1}\t{hit5}\t{hit10}\t{average_random_hit}"

    # turn info
    ave_turn, success_rate, ave_turn_done = cal_dialog_info(data)
    line = line + f"\t{ave_turn}\t{success_rate}\t{ave_turn_done}"
    # print(line)

    # count the number of entity plugin: AskForClarification in assistant's content and the last action is SearchNodes
    entity_counts = []
    for d in data:
        entity_count = 0
        for i in range(len(d["dialog"])):
            if d["dialog"][i]["role"] == "assistant":
                if (
                    "AskForClarification" in d["dialog"][i]["content"]
                    and "SearchNodes" in d["dialog"][i - 2]["content"]
                ):
                    entity_count += 1
        entity_counts.append(entity_count)

    ave_entity_count = sum(entity_counts) / len(entity_counts)

    # count the number of intention plugin: AskForClarification in assistant's content and the last action is SearchGraphPatterns
    intention_counts = []
    for d in data:
        intention_count = 0
        for i in range(len(d["dialog"])):
            if d["dialog"][i]["role"] == "assistant":
                if (
                    "AskForClarification" in d["dialog"][i]["content"]
                    and "SearchGraphPatterns" in d["dialog"][i - 2]["content"]
                ):
                    intention_count += 1
        intention_counts.append(intention_count)

    ave_intention_count = sum(intention_counts) / len(intention_counts)

    print(colorful(f"average_f1\tave_entity_count\tave_intention_count", "red"))
    print(f"{average_f1:.2f}\t{ave_entity_count:.2f}\t{ave_intention_count:.2f}")


if __name__ == "__main__":
    # python evaluation/eval_all.py --path save/v1/cwq/gpt-4o-2024-11-20
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default=None)
    args = parser.parse_args()

    # args.path = "save/v1/cwq/Meta-Llama-3.1-8B-Instruct-epoch3"

    if "wqcwq" in args.path:
        res = evaluation_wqcwq(args.path)
    elif "cwq" in args.path:
        res = evaluation_cwq(args.path)
    elif "webqsp" in args.path:
        res = evaluation_webqsp(args.path)
    else:
        raise ValueError(f"unknown dataset. You should specify cwq, webqsp or kqapro.")
    # print(res)
