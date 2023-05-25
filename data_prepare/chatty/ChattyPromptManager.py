import json
import pickle
import timeit

import numpy as np
import openai
import pandas as pd

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def analyze_with_backoff(comm):
    return analyze_text(comm)


def analyze_text(comment):
    if comment.strip() == "":
        return [-1, -1, -1]
    result = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "you are a helpful assistant that evaluates students emotional quotient, intelligence "
                        "quotient and academic progress based on comments about them. You can only answer with 3 "
                        "numbers from 0 to 5, where 0 means \"can not evaluate\", 1 means poor and 5 - excellent. You "
                        "always answer in format \"EQ: X, IQ: X, AP: X\", where X is a number. No additional comments "
                        "are needed."},
            {"role": "user", "content": generate_prompt(comment)}
        ],
        temperature=0,
        max_tokens=15,
        stop=["."]
    )
    numbers = result.choices[0].message.content.strip()
    answer = [x.strip() for x in numbers.split(',')]
    return answer


def generate_prompt(comment):
    return "{0}".format(
        comment
    )


def dump_results(log_path, result):
    file = open(log_path, 'wb')
    pickle.dump(result, file)
    file.close()


class ChattyPromptManager:
    def __init__(self, comments, log_path):
        self.parsed_result = None
        self.comments = comments
        self.log_path = log_path
        data = json.load(open("env.json", 'r'))
        openai.api_key = data['OPENAI_API_KEY']
        self.time = 0
        self.result = []

    def prepare_prompts_results(self):
        self.result = []
        start_time = timeit.default_timer()
        for comm in self.comments:
            if comm.strip() == "":
                self.result.append(["EQ: 0", "IQ: 0", "AP: 0"])
            else:
                self.result.append(analyze_with_backoff(comm))
        end_time = timeit.default_timer()
        self.time = end_time - start_time
        dump_results(self.log_path + "raw", self.result)
        self.parse_result()
        return self.parsed_result

    def parse_result(self):
        data = np.array(self.result, dtype=object)
        parsed = list(map(parse_chatty, data))
        parsed_pd = pd.DataFrame(parsed, columns=['EQ', 'IQ', 'AP', 'M'])
        parsed_pd.loc[parsed_pd['AP'] > 5, 'AP'] = 5
        parsed_pd.loc[parsed_pd['EQ'] > 5, 'EQ'] = 5
        parsed_pd.loc[parsed_pd['IQ'] > 5, 'IQ'] = 5
        self.parsed_result = parsed_pd
        dump_results(self.log_path + "parsed", self.parsed_result)


class Meaningfulness:
    def __init__(self):
        self.value = 0

    def increase(self):
        self.value += 1


def parse_chatty(lst):
    result = [0, 0, 0, 0]
    meaningfulness = Meaningfulness()
    if len(lst) < 3:
        return result
    extract_num(lst, 0, result, 'Emotional quotient:', meaningfulness)
    extract_num(lst, 1, result, 'Intelligence quotient:', meaningfulness)
    extract_num(lst, 2, result, 'Academic progress:', meaningfulness)
    extract_num(lst, 0, result, 'EQ:', meaningfulness)
    extract_num(lst, 1, result, 'IQ:', meaningfulness)
    extract_num(lst, 2, result, 'AP:', meaningfulness)
    result[3] = meaningfulness.value
    return result


def extract_num(lst, ind, result, string, meaningfulness):
    sz = len(string)
    if lst[ind][:sz] == string:
        result[ind] = convert_to_int(lst[ind][sz + 1:sz + 2])
        meaningfulness.increase()


def convert_to_int(value):
    if value is None:
        return 0
    try:
        float(value)
        return int(float(value))
    except:
        return 0
