"""Compare results/ with every exp*/expected.json and print PASS/FAIL per number.

A number passes when it rounds to the paper's printed value: the tolerance is
half a unit in the last digit the paper prints ("0.602" -> 0.0005). A path
ending in "se" on a list takes that list's standard error.

    python check_results.py
"""
import glob, json, statistics, sys


def lookup(path, keys):
    v = json.load(open(path))
    for k in keys:
        if k == "se" and isinstance(v, list):
            return statistics.stdev(v) / len(v) ** 0.5
        v = v[k]
    return v


fails = 0
for exp in sorted(glob.glob("exp*/expected.json")):
    print(f"\n{exp}")
    for e in json.load(open(exp)):
        want = e["expected"]
        tol = 0.5 * 10 ** -(len(want.split(".")[1]) if "." in want else 0) + 1e-9
        try:
            got = float(lookup(e["file"], e["path"]))
            ok = abs(got - float(want)) <= tol
            shown = f"{got:.6g}"
        except (OSError, KeyError, IndexError, TypeError, ValueError) as err:
            ok, shown = False, f"missing ({type(err).__name__})"
        fails += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {want:>7}  got {shown:<12} {e['paper']}")

print(f"\n{'ALL PASS' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)
