import numpy as np
import copy
import functools
import enum


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


def is_empty(set: set) -> bool:
    return len(set) == 0


def qiHelper(matrix: list[list[int]], i: int, prevQ: list[int]) -> list[int]:
    acc: list[int] = []
    diff = set(Si(matrix, i)).difference(set(Si(matrix, i-1)))

    for x in diff:
        if is_empty(set(upperSection(matrix, x)).intersection(set(prevQ))):
            acc.append(x)

    return acc


def Qi(matrix: list[list[int]], i: int) -> list[int]:
    if (i == 0):
        return S0(matrix)

    prevQ = Qi(matrix, i-1)
    return list(set(prevQ).union(set(qiHelper(matrix, i, prevQ))))


def neumann_morgenstern(matrix: list[list[int]], omegaPower: int) -> list[int]:
    omega = set(range(omegaPower - 1))
    i = 0
    while set(Si(matrix, i)) != omega:
        i += 1

    l = i

    result = Qi(matrix, l)
    return list(map(lambda x: x + 1, result))


# print(neumann_morgenstern(relations[0], 15))

# k-optimization

N = "N"
P = "P"
I = "I"
EMPTY = None


class PIN(enum.Enum):
    N = N
    P = P
    I = I
    EMPTY = EMPTY


def mirror_elem(matrix: list[list], x, y):
    return matrix[y][x]


def replace_symmetrical_elems(matrix: list[list[int]], elem_val, replacement):
    matrix_copy = copy.deepcopy(matrix)
    def applied_mirr_elem(x, y): return mirror_elem(matrix, x, y)

    for x, row in enumerate(matrix):
        for y, elem in enumerate(row):
            if elem == elem_val and applied_mirr_elem(x, y) == elem_val:
                matrix_copy[x][y] = replacement
                matrix_copy[y][x] = replacement

    return matrix_copy


def extractN(matrix): return replace_symmetrical_elems(matrix, 0, N)
def extractI(matrix): return replace_symmetrical_elems(matrix, 1, I)


def extractP(matrix: list[list[int]]):
    matrix_copy = copy.deepcopy(matrix)
    def applied_mirr_elem(x, y): return mirror_elem(matrix, x, y)

    for x, row in enumerate(matrix):
        for y, elem in enumerate(row):
            if elem == 1 and applied_mirr_elem(x, y) == 0:
                matrix_copy[x][y] = P
                matrix_copy[y][x] = EMPTY

            if elem == 0 and applied_mirr_elem(x, y) == 1:
                matrix_copy[x][y] = EMPTY
                matrix_copy[y][x] = P

    return matrix_copy

# the following snippet taken from https://stackoverflow.com/questions/16739290/composing-functions-in-python


def compose2(f, g):
    return lambda *a, **kw: f(g(*a, **kw))


def compose(*fs):
    return functools.reduce(compose2, fs)

# end of snippet


extractPIN = compose(extractP, extractI, extractN)

print(np.array(extractPIN(relations[2].tolist())))


def K(matrix: list[list[PIN]], vals_to_replace: list[PIN]) -> list[list[int]]:
    matrix_copy = copy.deepcopy(matrix)

    for x, row in enumerate(matrix):
        for y, elem in enumerate(row):
            if (elem in vals_to_replace):
                matrix_copy[x][y] = 1

    return matrix_copy


def K1(matrix: list[list[PIN]]): return K(matrix, [P, I, N])
def K2(matrix: list[list[PIN]]): return K(matrix, [P, N])
def K3(matrix: list[list[PIN]]): return K(matrix, [P, I])
def K4(matrix: list[list[PIN]]): return K(matrix, [P])


pin = extractPIN(relations[2].tolist())
