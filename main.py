import os
import sys

from src.write_utils import write_local_excel

DIR = "data"
FILE_NAME = "labels.xlsx"


def main() -> None:
    try:
        write_local_excel(DIR + "/" + FILE_NAME)
    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
