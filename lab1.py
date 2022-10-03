import numpy as np


def stringsToInts(arr: list[str]) -> list[int]:
    return list(map(lambda s: int(s), arr))


def parse_binary_relations(path: str, omegaPower: int) -> list[list[int]]:
    file = open(path, "r")
    lines = file.readlines()
    lines_without_separators = list(
        filter(lambda line: not line.startswith("R"), lines)
    )

    lines_without_escapes = list(map(
        lambda s: s.replace("\n", ""), lines_without_separators
    ))

    stripped_lines = list(map(lambda s: s.strip(), lines_without_escapes))
    splitted_lines = list(map(lambda s: s.split("  "), stripped_lines))
    numeric_lines = list(map(lambda arr: stringsToInts(arr), splitted_lines))

    number_of_matrixes: int = len(numeric_lines)/omegaPower
    relations = np.array_split(numeric_lines, number_of_matrixes)

    # for rel in relations:
    #     listy_rel = rel.tolist()
    #     print(is_acyclic(listy_rel))

    return relations


def remove_node(matrix: list[list[int]], i: int) -> list[list[int]]:
    matrix_without_row = matrix[:i] + matrix[i+1:]
    for row in matrix_without_row:
        row.pop(i)

    return matrix_without_row


def areZeros(lst: list[int]) -> bool:
    accumulator = False
    for elem in lst:
        accumulator = accumulator | elem

    return not accumulator


def index_of_first(lst, pred):
    for i, v in enumerate(lst):
        if pred(v):
            return i
    return None


def find_leaf(matrix: list[list[int]]) -> int:
    leaf_index = index_of_first(matrix, areZeros)
    return leaf_index


def is_acyclic(matrix: list[list[int]]) -> bool:
    if len(matrix) == 0:
        return True

    leaf_index: int | None = find_leaf(matrix)
    if (leaf_index == None):
        return False

    return is_acyclic(remove_node(matrix, leaf_index))


# Neuman-Morgenstern

def find_all_indexes(lst: list, pred) -> list[int]:
    indexes = []
    for i, v in enumerate(lst):
        if pred(v):
            indexes.append(i)

    return indexes


def upperSection(matrix: list[list[int]], i: int) -> list[int]:
    section = []
    for index, row in enumerate(matrix):
        if row[i] == 1:
            section.append(index)

    return section


def S0(matrix: list[list[int]]) -> list[int]:
    s0 = []
    for i, _ in enumerate(matrix):
        if areZeros(upperSection(matrix, i)):
            s0.append(i)

    return s0


def siHelper(matrix: list[list[int]], prevS: list[int]) -> list[int]:
    acc = []
    for i, _ in enumerate(matrix):
        if set(upperSection(matrix, i)).issubset(set(prevS)):
            acc.append(i)

    return acc


def Si(matrix: list[list[int]], i: int) -> list[int]:
    if i == 0:
        return S0(matrix)

    prevS = Si(matrix, i-1)
    return list(set(prevS).union(set(siHelper(matrix, prevS))))


relations = parse_binary_relations('./lab_2_variant_52.txt', 15)
