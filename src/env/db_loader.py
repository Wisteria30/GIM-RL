# -*- coding: utf-8 -*-

import numpy as np


def convert_list_str2float32(str_list):
    return list(map(np.float32, str_list.split()))


def clip_dataset(transaction, used=1.0, head=0.0, shuffle_db=False):
    head_point = int(len(transaction) * head)
    # If the range of the clip exceeds the size of the size, round it down.
    clip = min(int(len(transaction) * used + len(transaction) * head), len(transaction))
    if shuffle_db:
        np.random.shuffle(transaction)
    return transaction[head_point:clip]


def load_db(path, used, head, shuffle_db):
    with open(path) as f:
        transaction = f.readlines()

    items = set()
    transaction = clip_dataset(transaction, used, head, shuffle_db)
    transaction = np.array([list(map(np.float32, t.split())) for t in transaction])
    for t in transaction:
        items = items | set(t)

    b2i_dict = {i: v for i, v in enumerate(items)}
    i2b_dict = {v: i for i, v in enumerate(items)}
    bit_map = np.array(
        [[0 for _ in range(len(items))] for _ in range(len(transaction))], dtype=np.int8
    )

    for i, t in enumerate(transaction):
        for item in t:
            bit_map[i][i2b_dict[item]] = 1

    return transaction, bit_map, b2i_dict, i2b_dict


def load_hui_db(path, used, head, shuffle_db):
    with open(path) as f:
        transaction = f.readlines()

    # Database: Dictionary type transaction list of itemsets and utilities
    database = np.empty(0, dtype=np.float32)
    # items: Set of all items
    items = set()
    # utils: List of utility values for each transaction
    utils = np.empty(0, dtype=np.float32)

    transaction = clip_dataset(transaction, used, head, shuffle_db)
    for t in transaction:
        left, center, right = t.split(":")
        left = convert_list_str2float32(left)
        right = convert_list_str2float32(right)
        database = np.append(database, {left[i]: right[i] for i in range(len(left))})
        items = items | set(left)
        utils = np.append(utils, np.float32(center))

    return database, items, utils
