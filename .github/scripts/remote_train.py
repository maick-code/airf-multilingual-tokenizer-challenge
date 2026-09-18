"""Remote training experiments, executed on a GitHub-hosted runner.

Two jobs in one script:

1. Reproduce the submitted BPE tokenizer from a fresh Hub download of the
   official train split (revision v1.0.0) and assert byte-identity with the
   submitted artifact. This proves the recipe end-to-end outside the research
   sandbox, using no mirrored files.
2. Train an exploratory Unigram tokenizer with the same serialization and
   language weights, evaluate it on the full official validation split with
   the repository's own scoring code, and export the artifacts.

Everything runs offline from the pinned Hub dataset; no external corpora.
"""
from __future__ import annotations

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

from competition.data import Example, load_dataset  # noqa: E402
from competition.metrics import score_tokenizer, unknown_token_id  # noqa: E402

LANGUAGES = ("en", "fr", "ha", "sw", "yo", "am")
DATASET_ID = "Similoluwa/african-multilingual-tokenizer-challenge"
REVISION = "v1.0.0"
EXPECTED_TRAIN_JSONL_SHA256 = "85140695f33ff87ead838dbac449fbc7f13aec72dde4129556d7538ba17cbc4f"
EXPECTED_BPE_TOKENIZER_SHA256 = "4043a0499c1e4578bbc636a91f580abfe77dbe0ac52af4dbb0335743e973f5ed"
UNITS = {"en": 32, "fr": 41, "ha": 136, "sw": 240, "yo": 176, "am": 176}
DENOMINATOR = 32
OUT = ROOT / "remote_train_outputs"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def serialize_splits():
    """Round-robin train JSONL + source-order validation JSONL and CSV."""
    from datasets import load_dataset as hub_load

    OUT.mkdir(parents=True, exist_ok=True)
    train = hub_load(DATASET_ID, split="train", revision=REVISION)
    by_language = {lang: [] for lang in LANGUAGES}
    for row in train:
        language, text = row["language"], row["text"]
        assert language in LANGUAGES and isinstance(text, str) and text.split()
        by_language[language].append(text)
    assert all(len(items) == 40_000 for items in by_language.values())
    train_path = OUT / "train.jsonl"
    with train_path.open("w", encoding="utf-8", newline="\n") as output:
        for group in zip(*(by_language[lang] for lang in LANGUAGES), strict=True):
            for language, text in zip(LANGUAGES, group, strict=True):
                output.write(json.dumps([language, text], ensure_ascii=False, separators=(",", ":")) + "\n")

    validation = hub_load(DATASET_ID, split="validation", revision=REVISION)
    validation_path = OUT / "validation.jsonl"
    validation_csv = OUT / "validation.csv"
    counts = Counter()
    with validation_path.open("w", encoding="utf-8", newline="\n") as jsonl, \
            validation_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["language", "text"])
        for row in validation:
            language, text = row["language"], row["text"]
            assert language in LANGUAGES and isinstance(text, str) and text.split()
            counts[language] += 1
            jsonl.write(json.dumps([language, text], ensure_ascii=False, separators=(",", ":")) + "\n")
            writer.writerow([language, text])
    assert counts == dict.fromkeys(LANGUAGES, 4_000), dict(counts)

    train_sha = sha256_file(train_path)
    print("train.jsonl sha256:", train_sha)
    assert train_sha == EXPECTED_TRAIN_JSONL_SHA256, (
        "fresh Hub download serializes differently than the research fingerprint"
    )
    return train_path, validation_csv


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            language, text = json.loads(line)
            yield language, text


def weighted_texts(rows):
    seen = Counter()
    for language, text in rows:
        index = seen[language]
        seen[language] += 1
        copies = ((index + 1) * UNITS[language]) // DENOMINATOR - (index * UNITS[language]) // DENOMINATOR
        if copies < 1:
            raise ValueError("A train row was omitted")
        for _ in range(copies):
            yield text


def train_bpe(train_path, output):
    weighted_rows = sum(40_000 * UNITS[lang] // DENOMINATOR for lang in LANGUAGES)
    started = time.perf_counter()
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=10_000,
        min_frequency=5,
        special_tokens=[],
        initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()),
        show_progress=False,
    )
    tokenizer.train_from_iterator(weighted_texts(read_rows(train_path)), trainer=trainer, length=weighted_rows)
    tokenizer.save(str(output), pretty=True)
    return time.perf_counter() - started


def train_unigram(train_path, output):
    weighted_rows = sum(40_000 * UNITS[lang] // DENOMINATOR for lang in LANGUAGES)
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
    tokenizer.train_from_iterator(weighted_texts(read_rows(train_path)), trainer=trainer, length=weighted_rows)
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


def evaluate(tokenizer_path, validation_csv):
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    examples = load_dataset(validation_csv)
    result = score_tokenizer(tokenizer, examples, benchmark_repeats=3)
    texts = [item.text for item in examples]
    report = {
        "score": result.score,
        "fertility": result.fertility,
        "unknown_tokens": result.unknown_tokens,
        "unk_id_defined": unknown_token_id(tokenizer) is not None,
        "reconstruction": result.reconstruction,
        "guardrail_penalty": result.guardrail_penalty,
        "vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
        "strict_lossless_failures": strict_failure_count(tokenizer, texts),
        "elapsed_seconds": result.elapsed_seconds,
        "throughput": result.throughput,
        "sha256": sha256_file(tokenizer_path),
    }
    return report


def run_in_subprocess(label, train_fn_name, train_path, output):
    """Train in a child process so huge allocations are released."""
    code = (
        "import sys; sys.path.insert(0, %r); "
        "from pathlib import Path; "
        "import importlib.util; "
        "spec = importlib.util.spec_from_file_location('rt', %r); "
        "rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt); "
        f"seconds = rt.{train_fn_name}(Path({str(train_path)!r}), Path({str(output)!r})); "
        "print('TRAINING_SECONDS', seconds)"
    ) % (str(ROOT), str(Path(__file__).resolve()))
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if proc.returncode != 0:
        return {"status": "failed", "error": (proc.stderr or proc.stdout)[-3000:]}
    seconds = None
    for line in proc.stdout.splitlines():
        if line.startswith("TRAINING_SECONDS"):
            seconds = float(line.split()[1])
    print(f"[{label}] training done in {seconds:.1f}s")
    return {"status": "ok", "training_seconds": seconds}


def main():
    report = {"runner": os.environ.get("RUNNER_NAME", ""), "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    train_path, validation_csv = serialize_splits()
    report["train_jsonl_sha256"] = sha256_file(train_path)
    report["validation_csv_sha256"] = sha256_file(validation_csv)

    # 1. BPE reproduction of the submitted artifact.
    bpe_path = OUT / "bpe-repro-tokenizer.json"
    bpe = run_in_subprocess("bpe", "train_bpe", train_path, bpe_path)
    report["bpe"] = bpe
    if bpe["status"] == "ok":
        bpe_sha = sha256_file(bpe_path)
        report["bpe"]["sha256"] = bpe_sha
        report["bpe"]["matches_submission"] = bpe_sha == EXPECTED_BPE_TOKENIZER_SHA256
        report["bpe"]["evaluation"] = evaluate(bpe_path, validation_csv)

    # 2. Exploratory Unigram.
    uni_path = OUT / "unigram-tokenizer.json"
    uni = run_in_subprocess("unigram", "train_unigram", train_path, uni_path)
    report["unigram"] = uni
    if uni["status"] == "ok":
        report["unigram"]["evaluation"] = evaluate(uni_path, validation_csv)

    report_path = OUT / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
