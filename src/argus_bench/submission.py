"""Check an ARGUS submission (a folder or a .zip) and optionally run its verifier on its own rollouts.

    rollouts/*.jsonl                one file per rollout (argus-bench format)
    labels.json                     {"<rollout file name>": true | false}   true = contains the misbehavior
    verifier.py                     def verify(rollout: dict, artifacts_dir: Path, llm: OpenAI) -> Verdict
    verifier-requirements.txt       packages the verifier needs
    verifier_config.json            only if the verifier uses `llm`: {"model": "<hf id>", "revision": "<full hf commit hash>"}
    artifacts/                      anything else the verifier needs (suggested: artifacts/<rollout id>/... per rollout)
"""
import ast
import copy
import importlib.util
import json
import os
import sys
import tempfile
import zipfile

from .check import check, load
from .rollout import load_rollout
from .types import Verdict


def _root(path):
    """Directory holding rollouts/ (unzips a .zip into a temp dir; the zip may have one top-level folder)."""
    path = os.fspath(path)
    tmp = None
    if os.path.isfile(path) and path.endswith(".zip"):
        tmp = tempfile.TemporaryDirectory()
        zipfile.ZipFile(path).extractall(tmp.name)
        path = tmp.name
    if not os.path.isdir(os.path.join(path, "rollouts")):
        subs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d, "rollouts"))] if os.path.isdir(path) else []
        if len(subs) == 1:
            path = os.path.join(path, subs[0])
    return path, tmp


def _verifier_ast(path):
    tree = ast.parse(open(path).read())
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "verify"), None)
    uses_llm = False
    if fn is not None:
        names = [a.arg for a in fn.args.args]
        uses_llm = any(isinstance(n, ast.Name) and n.id == (names[2] if len(names) > 2 else "llm") for b in fn.body for n in ast.walk(b))
        uses_llm = uses_llm or any(isinstance(n, ast.Name) and n.id == "llm" for b in fn.body for n in ast.walk(b))
    return fn, uses_llm


def check_submission(path, run=False, llm_base_url=None, log=print):
    """Returns (problems, warnings, report). With run=True the verifier is executed on every rollout (it is the submitter's own code: only do this
    for code you trust) and its verdicts are compared with labels.json."""
    problems, warnings, report = [], [], {}
    root, tmp = _root(path)
    try:
        rdir = os.path.join(root, "rollouts")
        everything = sorted(f for f in os.listdir(rdir) if not f.startswith(".")) if os.path.isdir(rdir) else []
        files = [f for f in everything if f.endswith(".jsonl")]
        for f in everything:
            if f not in files:
                warnings.append(f"rollouts/{f}: not a .jsonl file, ignored")
        if not files:
            return ["no rollout files (*.jsonl) in rollouts/"], warnings, report
        # labels
        lp = os.path.join(root, "labels.json")
        labels = {}
        if not os.path.exists(lp):
            problems.append("labels.json is missing")
        else:
            try:
                labels = json.load(open(lp))
            except ValueError as e:
                problems.append(f"labels.json is not valid JSON ({e})")
        if labels:
            if not isinstance(labels, dict) or not all(isinstance(v, bool) for v in labels.values()):
                problems.append("labels.json must map each rollout file name to true or false")
                labels = {}
            else:
                for f in files:
                    if f not in labels:
                        problems.append(f"labels.json has no label for rollouts/{f}")
                for k in labels:
                    if k not in files:
                        problems.append(f"labels.json lists '{k}' but there is no such file in rollouts/")
        pos = sum(1 for f in files if labels.get(f) is True)
        neg = sum(1 for f in files if labels.get(f) is False)
        report["rollouts"], report["misbehavior"], report["benign"] = len(files), pos, neg
        if labels and (pos == 0 or neg == 0):
            warnings.append("only one kind of label: include both misbehavior and benign rollouts of the same task")
        # each rollout
        models = set()
        for f in files:
            try:
                p, w, info = check(os.path.join(rdir, f))
            except ValueError as e:
                problems.append(f"rollouts/{f}: {e}")
                continue
            problems += [f"rollouts/{f}: {x}" for x in p]
            warnings += [f"rollouts/{f}: {x}" for x in w]
            models |= set(info.get("models", []))
        if len(models) > 1:
            warnings.append(f"rollouts use several models {sorted(models)}; the form asks for one model per submission")
        report["models"] = sorted(models)
        # verifier files
        vp = os.path.join(root, "verifier.py")
        uses_llm = False
        if not os.path.exists(vp):
            problems.append("verifier.py is missing")
        else:
            try:
                fn, uses_llm = _verifier_ast(vp)
                if fn is None:
                    problems.append("verifier.py does not define `def verify(rollout, artifacts_dir, llm)`")
                elif len(fn.args.args) != 3:
                    problems.append("verify() must take exactly (rollout, artifacts_dir, llm)")
            except SyntaxError as e:
                problems.append(f"verifier.py has a syntax error: {e}")
        if not os.path.exists(os.path.join(root, "verifier-requirements.txt")):
            problems.append("verifier-requirements.txt is missing (an empty file is fine if nothing is needed)")
        cp = os.path.join(root, "verifier_config.json")
        if uses_llm and not os.path.exists(cp):
            problems.append("verify() uses `llm` but verifier_config.json is missing (it must name the model and revision the verifier calls)")
        if os.path.exists(cp):
            try:
                cfg = json.load(open(cp))
                for k in ("model", "revision"):
                    if not cfg.get(k):
                        problems.append(f'verifier_config.json needs "{k}"')
                if cfg.get("revision") and len(str(cfg["revision"])) != 40:
                    warnings.append("verifier_config.json: revision should be the full 40-character commit hash")
            except ValueError as e:
                problems.append(f"verifier_config.json is not valid JSON ({e})")
        elif not uses_llm:
            pass
        report["verifier_uses_llm"] = uses_llm
        # run
        if run and not problems:
            report.update(_run(root, files, labels, llm_base_url, uses_llm, warnings, log))
        return problems, warnings, report
    finally:
        if tmp:
            tmp.cleanup()


def _run(root, files, labels, llm_base_url, uses_llm, warnings, log):
    sys.path.insert(0, root)
    try:
        spec = importlib.util.spec_from_file_location("argus_submission_verifier", os.path.join(root, "verifier.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    llm = None
    if llm_base_url:
        import openai
        llm = openai.OpenAI(base_url=llm_base_url, api_key=os.environ.get("OPENAI_API_KEY", "none"))
    elif uses_llm:
        warnings.append("the verifier uses `llm` but no --llm-base-url was given: it receives llm=None")
    from pathlib import Path
    results, errors = [], []
    for f in files:
        rollout = load_rollout(os.path.join(root, "rollouts", f))
        try:
            v = mod.verify(copy.deepcopy(rollout), Path(root) / "artifacts", llm)
            if isinstance(v, dict):
                v = Verdict.from_dict(v)
            if not hasattr(v, "misbehavior"):
                raise TypeError(f"verify() returned {type(v).__name__}, expected argus_bench.Verdict")
            results.append((f, bool(v.misbehavior), labels[f], v))
            log(f"  {f}: verdict={bool(v.misbehavior)} label={labels[f]} {'ok' if bool(v.misbehavior) == labels[f] else 'MISMATCH'}")
        except Exception as e:
            errors.append(f"{f}: {type(e).__name__}: {e}")
            log(f"  {f}: ERROR {type(e).__name__}: {e}")
    tp = sum(1 for _, v, l, _ in results if v and l)
    tn = sum(1 for _, v, l, _ in results if not v and not l)
    fp = sum(1 for _, v, l, _ in results if v and not l)
    fn = sum(1 for _, v, l, _ in results if not v and l)
    n = len(results)
    return {"verifier_run": {"rollouts_run": n, "errors": errors, "accuracy": (tp + tn) / n if n else None,
                              "true_positive": tp, "true_negative": tn, "false_positive": fp, "false_negative": fn,
                              "missing_explanation": sum(1 for _, _, _, v in results if not v.explanation)}}
