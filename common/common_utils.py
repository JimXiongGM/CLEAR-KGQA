import argparse
import functools
import json
import os
import pickle
import re
import subprocess
import traceback
from concurrent import futures
from datetime import date, datetime

from tqdm import tqdm


def read_json(path="test.json"):
    with open(path, "r", encoding="utf-8") as f1:
        res = json.load(f1)
    return res


class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(obj, date):
            return obj.strftime("%Y-%m-%d")
        else:
            return json.JSONEncoder.default(self, obj)


def _set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


def replace_ignorecase(text, _from, _to):
    try:
        res = re.sub(str(_from), str(_to), str(text), flags=re.I)
    except Exception as e:
        # print(f"\nerror from `{_from}` to `{_to}`.")
        res = text.replace(_from, _to)
    return res


def save_to_json(obj, path, _print=True):
    if _print:
        print(f"SAVING: {path}")
    if type(obj) == set:
        obj = list(obj)
    dirname = os.path.dirname(path)
    if dirname and dirname != ".":
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f1:
        json.dump(
            obj,
            f1,
            ensure_ascii=False,
            indent=4,
            cls=ComplexEncoder,
            default=_set_default,
        )
    if _print:
        res = subprocess.check_output(f"ls -lh '{path}'", shell=True).decode(encoding="utf-8")
        print(res)


def read_pkl(path="test.pkl"):
    with open(path, "rb") as f1:
        res = pickle.load(f1)
    return res


def save_to_pkl(obj, path, _print=True):
    dirname = os.path.dirname(path)
    if dirname and dirname != ".":
        os.makedirs(dirname, exist_ok=True)
    with open(path, "wb") as f1:
        pickle.dump(obj, f1)
    if _print:
        res = subprocess.check_output(f"ls -lh '{path}'", shell=True).decode(encoding="utf-8")
        print(res)


def read_jsonl(path="test.jsonl", desc="", max_instances=None, _id_to_index_key=False):
    with open(path, "r", encoding="utf-8") as f1:
        res = []
        _iter = tqdm(enumerate(f1), desc=desc, ncols=150) if desc else enumerate(f1)
        for idx, line in _iter:
            if max_instances and idx >= max_instances:
                break
            res.append(json.loads(line.strip()))
    if _id_to_index_key:
        id_to_index = {i[_id_to_index_key]: idx for idx, i in enumerate(res)}
        return res, id_to_index
    else:
        return res


def colorful(text, color="yellow"):
    if color == "yellow":
        text = "\033[1;33m" + str(text) + "\033[0m"
    elif color == "grey":
        text = "\033[1;30m" + str(text) + "\033[0m"
    elif color == "green":
        text = "\033[1;32m" + str(text) + "\033[0m"
    elif color == "red":
        text = "\033[1;31m" + str(text) + "\033[0m"
    elif color == "blue":
        text = "\033[1;94m" + str(text) + "\033[0m"
    else:
        pass
    return text


def underline_text(text):
    underline_seq = "\033[4m"  # ANSI escape sequence for underline
    reset_seq = "\033[0m"  # ANSI escape sequence to reset formatting

    return f"{underline_seq}{text}{reset_seq}"


def make_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="data", type=str)
    parser.add_argument("--load_model", action="store_true")
    parser.set_defaults(load_model=True)

    args = parser.parse_args()
    return args


def time_now(fotmat="%Y-%m-%d %H:%M:%S"):
    date_time = datetime.now().strftime(fotmat)
    return date_time


def multi_process(
    items,
    process_function,
    postprocess_function=None,
    total=None,
    cpu_num=None,
    chunksize=1,
    tqdm_disable=False,
    debug=False,
    spawn=False,
    dummy=False,
    unordered=False,
    **kwargs,
):

    if isinstance(items, list):
        total = len(items)

    import functools
    from multiprocessing import cpu_count

    mapper = functools.partial(process_function, **kwargs)

    cpu_num = 1 if debug else cpu_num
    cpu_num = cpu_num or cpu_count()

    # debug
    if debug:
        res = []
        for idx, i in tqdm(enumerate(items), ncols=100):
            r = mapper(i)
            if postprocess_function:
                r = postprocess_function(r)
            res.append(r)
        return res

    if dummy:
        from multiprocessing.dummy import Pool

        pool = Pool(processes=cpu_num)
    else:
        if spawn:
            import torch

            ctx = torch.multiprocessing.get_context("spawn")
            pool = ctx.Pool(processes=cpu_num)
        else:
            from multiprocessing import Pool

            pool = Pool(processes=cpu_num)

    res = []
    _d = "Dummy " if dummy else ""
    pbar = tqdm(
        total=total,
        ncols=100,
        colour="green",
        desc=f"{_d}{cpu_num} CPUs processing",
        disable=tqdm_disable,
    )

    if unordered:
        _func = pool.imap_unordered
    else:
        _func = pool.imap

    # if error, save to some tmp file.
    try:
        for r in _func(mapper, items, chunksize=chunksize):
            if postprocess_function:
                r = postprocess_function(r)
            res.append(r)
            pbar.update()
    except Exception as e:
        traceback.print_exc()
        _time = time_now()
        pickle.dump(res, open(f"error-save {_time}.pkl", "wb"))
        print(f"file save to: error-save {_time}.pkl")

    if res:
        return res


def timeout(seconds):
    executor = futures.ThreadPoolExecutor(1)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            future = executor.submit(func, *args, **kw)
            return future.result(timeout=seconds)

        return wrapper

    return decorator


def wc_l(path):
    try:
        res = subprocess.check_output(f"wc -l {path}", shell=True).decode(encoding="utf-8")
        line_num = int(res.split()[0])
    except Exception as e:
        line_num = None
    return line_num


# @timeout(10)
def file_line_count(path):
    return wc_l(path)


def time_diff_to_str(time_diff):
    hours = int(time_diff // 3600)
    minutes = int((time_diff % 3600) // 60)
    seconds = int(time_diff % 60)
    time_diff_str = f"{hours} hours {minutes} minutes {seconds} seconds"
    return time_diff_str


def exception_logger(name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as ex:
                # Ensure directory exists
                os.makedirs("log_exception", exist_ok=True)

                # Formulate the log filename
                log_filename = f"log_exception/{name}.log"

                # Get current timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Retrieve the traceback information
                tb = traceback.extract_tb(ex.__traceback__)

                # Extract the relevant data from the traceback
                file_name, line_number, function_name, text = tb[-1]

                # Collect log messages
                log_messages = [
                    f"{timestamp} ERROR Exception occurred in function '{func.__name__}':",
                    f"{timestamp} ERROR Exception type: {type(ex).__name__}",
                    f"{timestamp} ERROR Exception args: {ex.args}",
                    f"{timestamp} ERROR Occurred at: {file_name}, line {line_number}, in {function_name}",
                    f"{timestamp} ERROR Code context: {text}",
                    f"{timestamp} ERROR Function arguments: args={args}, kwargs={kwargs}",
                    "",
                ]

                # Write log messages to the file
                with open(log_filename, "a") as f:
                    for message in log_messages:
                        print(message, file=f)

                # Optionally re-raise the exception if you want to handle it further up
                raise

        return wrapper

    return decorator
