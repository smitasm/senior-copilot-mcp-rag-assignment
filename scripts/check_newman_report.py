"""Strict verdict for a Newman JSON report.

Newman only fails on *assertions*, and many requests in the provided Postman
collections have none - so a 404 or 500 on them would pass silently. This
checker also requires every request to return HTTP 200.

Usage:
    newman run <collection> --reporters json --reporter-json-export report.json
    python scripts/check_newman_report.py report.json [more_reports.json ...]
Exit code 0 = all good, 1 = at least one problem.
"""

import json
import sys


def check(path: str) -> list[str]:
    run = json.load(open(path, encoding="utf-8"))["run"]
    problems = []
    for ex in run["executions"]:
        name = ex["item"]["name"]
        resp = ex.get("response")
        if not resp:
            problems.append(f"{name}: no response ({ex.get('requestError')})")
        elif resp["code"] != 200:
            problems.append(f"{name}: HTTP {resp['code']}")
        for a in ex.get("assertions", []):
            if a.get("error"):
                problems.append(f"{name}: assertion '{a['assertion']}' failed: {a['error']['message']}")
    print(f"{path}: {len(run['executions'])} requests, "
          f"{run['stats']['assertions']['total']} assertions, {len(problems)} problem(s)")
    return problems


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    all_problems = [p for report in sys.argv[1:] for p in check(report)]
    for p in all_problems:
        print("  FAIL", p)
    sys.exit(1 if all_problems else 0)