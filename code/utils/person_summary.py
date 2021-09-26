from tabulate import tabulate
import numpy as np

train = {
    0: 20,
    1: 9,
    2: 2,
    3: 9,
    4: 3,
    5: 3,
    6: 7,
    9: 3,
    12: 9,
    13: 3,
    14: 9,
    15: 14,
    16: 9,
    17: 5,
    19: 3,
    24: 4,
    27: 9,
    29: 9,
    32: 5,
    33: 23,
}

valid = {
    0: 5,
    1: 2,
    3: 1,
    4: 1,
    6: 1,
    9: 2,
    13: 1,
    15: 2,
    16: 1,
    24: 1,
    31: 2,
    33: 4,
}

test = {
    7: 14,
    8: 16,
    10: 1,
    11: 6,
    20: 4,
    21: 3,
    22: 7,
    25: 3,
    26: 1,
    28: 1,
}

distinct_persons = set(list(train.keys()) + list(valid.keys()) + list(test.keys()))

n_persons = len(distinct_persons)



# samples per person
count = {}

def count_samples(ds):
    for k, v in ds.items():
        if k not in count:
            count[k] = 0

        count[k] += v

count_samples(train)
count_samples(valid)
count_samples(test)

max_ = max(count.values())
min_ = min(count.values())
mean = np.mean(list(count.values()))


pretty = [
    ["NPersons", n_persons],
    ["Min", min_],
    ["Max", max_],
    ["Mean", mean]
]
print(tabulate(pretty))
