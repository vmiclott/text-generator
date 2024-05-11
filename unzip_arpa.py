import gzip
import sys


def main(in_filename: str, out_filename: str) -> None:
    with gzip.open(in_filename, "rt") as input, open(out_filename, "w") as output:
        for line in input:
            output.write(line)


if __name__ == "__main__":
    in_filename = sys.argv[1]
    out_filename = sys.argv[2]
    main(in_filename, out_filename)
