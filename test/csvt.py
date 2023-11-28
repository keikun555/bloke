"""Transposes brench outputs"""

import csv
import os
import sys
from collections import defaultdict

if __name__ == "__main__":
    f = csv.DictReader(sys.stdin)
    output_csv: list[dict[str, int | str]] = []
    fieldnames: list[str] = ["benchmark"]
    for row in f:
        if len(output_csv) <= 0 or row["benchmark"] != output_csv[-1]["benchmark"]:
            output_csv.append(defaultdict(int))
            output_csv[-1]["benchmark"] = row["benchmark"]
        output_csv[-1][row["run"]] = row["result"]
        if row["run"] not in fieldnames:
            fieldnames.append(row["run"])

    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(output_csv)
