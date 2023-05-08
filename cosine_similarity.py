import math
import re
from collections import Counter

WORD = re.compile(r"\w+")

def calculate_cosine_similarity(text1, text2):
    if isinstance(text1, str):
        vec1 = text_to_vector(text1)
    else:
        vec1 = Counter(text1)
    if isinstance(text2, str):
        vec2 = text_to_vector(text2)
    else:
        vec2 = Counter(text2)

    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)