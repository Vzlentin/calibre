"""Fail unless every listed contract test passed, with no skip and no expected failure.

python .github/scripts/contract_report.py contract.xml tests/contract/ids.txt
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def node_id(classname: str, name: str) -> str:
    parts = classname.split(".")
    for cut in range(len(parts), 0, -1):
        path = Path(*parts[:cut]).with_suffix(".py")
        if path.exists():
            return "::".join([path.as_posix(), *parts[cut:], name])
    raise ValueError(f"no test file for {classname}")


def main(report: Path, ids: Path) -> int:
    cases = ET.parse(report).getroot().iter("testcase")
    passed, problems = set(), []
    for case in cases:
        test = node_id(case.get("classname", ""), case.get("name", ""))
        outcome = [child.tag for child in case if child.tag in ("failure", "error", "skipped")]
        if outcome:
            problems.append(f"{test}: {outcome[0]}")
        else:
            passed.add(test)
    expected = [line.strip() for line in ids.read_text().splitlines() if line.strip()]
    problems += [f"{test}: did not pass" for test in expected if test not in passed]
    for problem in problems:
        print(problem)
    print(f"{len(passed)} passed, {len(expected)} listed, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
