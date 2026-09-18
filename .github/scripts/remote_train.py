"""Remote training experiments, executed on a GitHub-hosted runner.

Parameterised through the command line so several configs run as parallel CI
jobs:

    python .github/scripts/remote_train.py --tag TAG --units JSON --denominator N

Trains a Unigram tokenizer on the official train split (fresh Hub download,
revision v1.0.0, round-robin serialization, cumulative weights), evaluates it
on the full official validation split with the repository's own scoring code,
and writes artifacts to remote_train_outputs/<tag>/.

Everything runs offline from the pinned Hub dataset; no external corpora.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402

from competition.data import load_dataset  # noqa: E402
from competition.metrics import score_tokenizer, unknown_token_id  # noqa: E402

LANGUAGES = ("en", "fr", "ha", "sw", "yo", "am")
DATASET_ID = "Similoluwa/african-multilingual-tokenizer-challenge"
REVISION = "v1.0.0"
EXPECTED_TRAIN_JSONL_SHA256 = "85140695f33ff87ead838dbac449fbc7f13aec72dde4129556d7538ba17cbc4f"
OUT_ROOT = ROOT / "remote_train_outputs"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def serialize_splits(out_dir):
    """Round-robin train JSONL + source-order validation JSONL and CSV."""
    from datasets import load_dataset as hub_load

    out_dir.mkdir(parents=True, exist_ok=True)
    train = hub_load(DATASET_ID, split="train", revision=REVISION)
    by_language = {lang: [] for lang in LANGUAGES}
    for row in train:
        language, text = row["language"], row["text"]
        assert language in LANGUAGES and isinstance(text, str) and text.split()
        by_language[language].append(text)
    assert all(len(items) == 40_000 for items in by_language.values())
    train_path = out_dir / "train.jsonl"
    with train_path.open("w", encoding="utf-8", newline="\n") as output:
        for group in zip(*(by_language[lang] for lang in LANGUAGES), strict=True):
            for language, text in zip(LANGUAGES, group, strict=True):
                output.write(json.dumps([language, text], ensure_ascii=False, separators=(",", ":")) + "\n")

    validation = hub_load(DATASET_ID, split="validation", revision=REVISION)
    validation_csv = out_dir / "validation.csv"
    counts = Counter()
    with validation_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["language", "text"])
        for row in validation:
            language, text = row["language"], row["text"]
            assert language in LANGUAGES and isinstance(text, str) and text.split()
            counts[language] += 1
            writer.writerow([language, text])
    assert counts == dict.fromkeys(LANGUAGES, 4_000), dict(counts)

    train_sha = sha256_file(train_path)
    print("train.jsonl sha256:", train_sha, flush=True)
    assert train_sha == EXPECTED_TRAIN_JSONL_SHA256, (
        "fresh Hub download serializes differently than the research fingerprint"
    )
    return train_path, validation_csv


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            language, text = json.loads(line)
            yield language, text


def weighted_texts(rows, units, denominator):
    seen = Counter()
    for language, text in rows:
        index = seen[language]
        seen[language] += 1
        copies = ((index + 1) * units[language]) // denominator - (index * units[language]) // denominator
        if copies < 1:
            raise ValueError("A train row was omitted")
        for _ in range(copies):
            yield text


def train_unigram(train_path, output, units, denominator):
    weighted_rows = sum(40_000 * units[lang] // denominator for lang in LANGUAGES)
    started = time.perf_counter()
    tokenizer = Tokenizer(models.Unigram())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.UnigramTrainer(
        vocab_size=10_000,
        special_tokens=[],
        initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()),
        max_piece_length=16,
        show_progress=False,
    )
    tokenizer.train_from_iterator(
        weighted_texts(read_rows(train_path), units, denominator),
        trainer=trainer,
        length=weighted_rows,
    )
    tokenizer.save(str(output), pretty=True)
    return time.perf_counter() - started


def strict_failure_count(tokenizer, texts, batch_size=1024):
    failures = 0
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        ids = [e.ids for e in tokenizer.encode_batch(batch, add_special_tokens=False)]
        dec_false = tokenizer.decode_batch(ids, skip_special_tokens=False)
        dec_true = tokenizer.decode_batch(ids, skip_special_tokens=True)
        failures += sum((t != a) or (t != b) for t, a, b in zip(batch, dec_false, dec_true, strict=True))
    return failures


def median_encode_seconds(tokenizer, texts, repeats=3):
    tokenizer.encode_batch(texts, add_special_tokens=False)
    timings = []
    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        tokenizer.encode_batch(texts, add_special_tokens=False)
        timings.append(time.perf_counter() - started)
    return sorted(timings)[len(timings) // 2]


def evaluate(tokenizer_path, validation_csv):
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    examples = load_dataset(validation_csv)
    texts = [item.text for item in examples]
    result = score_tokenizer(tokenizer, examples, benchmark_repeats=3)
    baseline_tokenizer = Tokenizer.from_file(str(ROOT / "starter/baselines/character-level/tokenizer.json"))
    baseline_seconds = median_encode_seconds(baseline_tokenizer, texts)
    candidate_seconds = median_encode_seconds(tokenizer, texts)
    return {
        "score": result.score,
        "fertility": result.fertility,
        "unknown_tokens": result.unknown_tokens,
        "unk_id_defined": unknown_token_id(tokenizer) is not None,
        "reconstruction": result.reconstruction,
        "guardrail_penalty": result.guardrail_penalty,
        "vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
        "strict_lossless_failures": strict_failure_count(tokenizer, texts),
        "elapsed_seconds": candidate_seconds,
        "baseline_seconds": baseline_seconds,
        "runtime_ratio_vs_char_baseline": candidate_seconds / baseline_seconds,
        "throughput": result.throughput,
        "sha256": sha256_file(tokenizer_path),
    }


def run_training_in_subprocess(train_path, output, units, denominator):
    code = (
        "import sys; sys.path.insert(0, %r); "
        "import importlib.util, json; "
        "from pathlib import Path; "
        "spec = importlib.util.spec_from_file_location('rt', %r); "
        "rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt); "
        "seconds = rt.train_unigram(Path(%r), Path(%r), json.loads(%r), %d); "
        "print('TRAINING_SECONDS', seconds)"
    ) % (
        str(ROOT), str(Path(__file__).resolve()), str(train_path), str(output),
        json.dumps(units), denominator,
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout)[-3000:])
    for line in proc.stdout.splitlines():
        if line.startswith("TRAINING_SECONDS"):
            return float(line.split()[1])
    raise RuntimeError("no training time reported")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--units", required=True, help="JSON dict of per-language integer units")
    parser.add_argument("--denominator", type=int, required=True)
    args = parser.parse_args()
    units = json.loads(args.units)
    assert set(units) == set(LANGUAGES) and all(
        isinstance(value, int) and value >= args.denominator for value in units.values()
    ), units
    weighted_rows = sum(40_000 * units[lang] // args.denominator for lang in LANGUAGES)

    out_dir = OUT_ROOT / args.tag
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    work_dir = OUT_ROOT / "data"
    train_path, validation_csv = serialize_splits(work_dir)

    tokenizer_path = out_dir / "tokenizer.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    seconds = run_training_in_subprocess(train_path, tokenizer_path, units, args.denominator)
    print(f"[{args.tag}] training done in {seconds:.1f}s", flush=True)

    report = {
        "tag": args.tag,
        "started_at": started_at,
        "runner": os.environ.get("RUNNER_NAME", ""),
        "config": {
            "model": "unigram",
            "units": units,
            "denominator": args.denominator,
            "vocab_size": 10_000,
            "max_piece_length": 16,
            "weighted_rows": weighted_rows,
        },
        "train_jsonl_sha256": sha256_file(train_path),
        "training_seconds": seconds,
        "evaluation": evaluate(tokenizer_path, validation_csv),
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
