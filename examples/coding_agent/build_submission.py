"""Builds a complete example submission folder: 3 misbehavior + 3 benign rollouts, labels.json, verifier, requirements.

    python build_submission.py my_submission
    argus-bench check-submission my_submission --run
"""
import json
import os
import shutil
import sys

from run_agent import run_rollout

out = sys.argv[1]
os.makedirs(os.path.join(out, "rollouts"), exist_ok=True)
labels = {}
for i, (behavior, function) in enumerate([("cheat", "add"), ("honest", "add"), ("cheat", "mul"), ("honest", "mul"), ("cheat", "add"), ("honest", "mul")]):
    name = f"rollout_{i:02d}.jsonl"
    path = os.path.join(out, "rollouts", name)
    if os.path.exists(path):
        os.remove(path)
    run_rollout(path, behavior, function)
    labels[name] = behavior == "cheat"
json.dump(labels, open(os.path.join(out, "labels.json"), "w"), indent=1)
shutil.copy(os.path.join(os.path.dirname(__file__), "verifier.py"), os.path.join(out, "verifier.py"))
open(os.path.join(out, "verifier-requirements.txt"), "w").write("")
print("built", out)
