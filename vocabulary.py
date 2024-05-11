import os
import string
import sys


def normalize(text: str) -> str:
    for punct in string.punctuation:
        text = text.replace(punct, "")
    return text.strip().lower()


def main(in_filename: str, out_filename: str) -> None:
    os.makedirs(os.path.dirname(out_filename), exist_ok=True)
    with open(in_filename, "r") as input, open(out_filename, "w") as output:
        vocabulary = set()
        line = input.readline()
        while line:
            normalized_line = normalize(line)
            parts = normalized_line.split()
            vocabulary.update(parts)
            line = input.readline()
        sorted_vocabulary = sorted(list(vocabulary))
        for word in sorted_vocabulary:
            output.write(word + "\n")


if __name__ == "__main__":
    in_filename = sys.argv[1]
    out_filename = sys.argv[2]
    main(in_filename, out_filename)
