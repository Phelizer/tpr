from dataclasses import replace
import numpy as np


def stringsToInts(arr: list[str]) -> list[int]:
    return list(map(lambda s: int(s), arr))


def parse_binary_relations(path: str, omegaPower: int):
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


parse_binary_relations('./lab_2_variant_52.txt', 15)
