"""Submission checker for the AI Research Foundations Multilingual
Tokenization Challenge.

This is the only helper file participants need. Download it next to your
notebook and run it on the ``tokenizer.json`` you are about to submit::

    from utils import profile_submission

    profile_submission("tokenizer.json", data=validation)

It applies the same rules as official evaluation: the vocabulary limit, the
coverage requirement, exact reconstruction, and the English and French
guardrail. Everything is local, so the file works unchanged in Google Colab,
on Kaggle, or in a clone of the competition repository.
"""

from __future__ import annotations

import json
import unicodedata
import statistics
import time
from collections import defaultdict
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers import __version__ as installed_tokenizers_version

__all__ = ["profile_submission"]

LANGUAGES = ("en", "fr", "ha", "sw", "yo", "am")
LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French",
    "ha": "Hausa",
    "sw": "Swahili",
    "yo": "Yoruba",
    "am": "Amharic",
}
SCORED_LANGUAGES = ("ha", "sw", "yo", "am")
CONTEXT_LANGUAGES = ("en", "fr")
CONTEXT_FERTILITY_RATIO = 1.15
RECONSTRUCTION_PENALTY = 3.0
UNKNOWN_PENALTY = 100.0
MAX_VOCAB_SIZE = 10_000
MAX_TOKENIZER_BYTES = 20 * 1024 * 1024
REQUIRED_TOKENIZERS_VERSION = "0.22.1"
SMOKE_TEXTS = {
    "en": "Knowledge grows when it is shared.",
    "fr": "Le savoir grandit lorsqu’il est partagé.",
    "ha": "Ilimi yana ƙaruwa idan an raba shi.",
    "sw": "Maarifa hukua yanaposhirikishwa.",
    "yo": "Ìmọ̀ ń pọ̀ sí i nígbà tí a bá pín in.",
    "am": "እውቀት ሲካፈል ያድጋል።",
}


def _rows(data) -> list[tuple[str, str]]:
    """Coerce supported containers into ``(language, text)`` pairs."""
    if hasattr(data, "itertuples"):
        return [(row.language, row.text) for row in data.itertuples()]
    if hasattr(data, "column_names"):
        return list(zip(data["language"], data["text"], strict=True))
    pairs: list[tuple[str, str]] = []
    for item in data:
        if isinstance(item, dict):
            pairs.append((item["language"], item["text"]))
        elif hasattr(item, "language") and hasattr(item, "text"):
            pairs.append((item.language, item.text))
        else:
            language, text = item
            pairs.append((language, text))
    return pairs


def _unknown_token_id(tokenizer: Tokenizer) -> int | None:
    """Return the id the tokenizer emits for text it cannot represent."""
    model = json.loads(tokenizer.to_str()).get("model", {})
    name = model.get("unk_token")
    if isinstance(name, str):
        return tokenizer.token_to_id(name)
    unk_id = model.get("unk_id")
    return int(unk_id) if unk_id is not None else None


def _measure(
    tokenizer: Tokenizer, rows: list[tuple[str, str]]
) -> tuple[dict, dict, int]:
    """Return fertility and unknown rate by language, plus lossy row count."""
    tokens: defaultdict = defaultdict(int)
    words: defaultdict = defaultdict(int)
    unknowns: defaultdict = defaultdict(int)
    unknown_id = _unknown_token_id(tokenizer)
    lossy = 0

    texts = [text for _, text in rows]
    encodings = tokenizer.encode_batch(texts, add_special_tokens=False)
    for (language, text), encoding in zip(rows, encodings, strict=True):
        tokens[language] += len(encoding.ids)
        words[language] += len(text.split())
        if unknown_id is not None:
            unknowns[language] += sum(
                1 for value in encoding.ids if value == unknown_id
            )
        restored = tokenizer.decode(encoding.ids, skip_special_tokens=True)
        if (unicodedata.normalize("NFC", restored).strip()
                != unicodedata.normalize("NFC", text).strip()):
            lossy += 1

    fertility = {
        language: tokens[language] / words[language]
        for language in LANGUAGES
        if words[language]
    }
    unknown_rate = {
        language: unknowns[language] / words[language]
        for language in LANGUAGES
        if words[language]
    }
    return fertility, unknown_rate, lossy


def _benchmark(tokenizer: Tokenizer, rows, repeats: int = 3) -> tuple[float, float]:
    """Return characters per second and the median elapsed seconds."""
    texts = [text for _, text in rows]
    total = sum(len(text) for text in texts)
    tokenizer.encode_batch(texts, add_special_tokens=False)
    timings = []
    for _ in range(max(1, repeats)):
        started = time.perf_counter()
        tokenizer.encode_batch(texts, add_special_tokens=False)
        timings.append(time.perf_counter() - started)
    elapsed = statistics.median(timings)
    return total / max(elapsed, 1e-12), elapsed


def _validate(path: Path) -> tuple[Tokenizer | None, dict, list[str], int | None]:
    """Apply the file-level validity contract to one tokenizer."""
    checks: dict[str, bool] = {}
    errors: list[str] = []

    if not path.is_file():
        return None, {"loads": False}, [f"{path} was not found"], None
    if path.name != "tokenizer.json":
        errors.append("the submitted file must be named tokenizer.json")

    checks["file_size"] = path.stat().st_size <= MAX_TOKENIZER_BYTES
    if not checks["file_size"]:
        limit = MAX_TOKENIZER_BYTES // 1_048_576
        errors.append(f"tokenizer.json is larger than {limit} MiB")
        return None, checks, errors, None

    try:
        tokenizer = Tokenizer.from_file(str(path))
        checks["loads"] = True
    except Exception as error:
        checks["loads"] = False
        errors.append(f"tokenizer could not be loaded: {error}")
        return None, checks, errors, None

    vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
    checks["vocabulary"] = vocab_size <= MAX_VOCAB_SIZE
    if not checks["vocabulary"]:
        errors.append(
            f"vocabulary has {vocab_size:,} entries; "
            f"the limit is {MAX_VOCAB_SIZE:,}"
        )

    smoke = list(SMOKE_TEXTS.items())
    fertility, _, _ = _measure(tokenizer, smoke)
    checks["encoding"] = len(fertility) == len(LANGUAGES)
    if not checks["encoding"]:
        errors.append("at least one language produced no tokens")

    checks["compatibility"] = (
        installed_tokenizers_version == REQUIRED_TOKENIZERS_VERSION
    )
    if not checks["compatibility"]:
        errors.append(
            f"expected tokenizers {REQUIRED_TOKENIZERS_VERSION} but found "
            f"{installed_tokenizers_version}"
        )
    return tokenizer, checks, errors, vocab_size


def _leader(label: str, width: int = 26) -> str:
    """Format a label padded with dots so report columns line up."""
    return f"{label}{'.' * max(1, width - len(label))}"


def profile_submission(
    path: str | Path = "tokenizer.json",
    data=None,
    *,
    repeats: int = 3,
    verbose: bool = True,
) -> dict:
    """Check, score and benchmark a tokenizer before you submit it.

    Args:
        path: Location of the ``tokenizer.json`` file to check.
        data: Optional labelled text used for the score, the coverage check and
            the benchmark. Any container of ``language`` and ``text`` pairs
            works. The validation split is the usual choice.
        repeats: Number of timed encoding passes; the median is reported.
        verbose: Print the human readable report as well as returning it.

    Returns:
        A dictionary with the checks, any errors, the vocabulary size and,
        when ``data`` is supplied, per language fertility, the competition
        score, unknown token count and throughput.
    """
    path = Path(path)
    tokenizer, checks, errors, vocab_size = _validate(path)
    report = {
        "path": str(path),
        "checks": checks,
        "errors": errors,
        "vocab_size": vocab_size,
        "tokenizers_version": installed_tokenizers_version,
    }

    if tokenizer is not None and data is not None:
        rows = _rows(data)
        fertility, unknown_rate, lossy = _measure(tokenizer, rows)
        penalised = {
            language: value + UNKNOWN_PENALTY * unknown_rate.get(language, 0.0)
            for language, value in fertility.items()
        }
        raw = sum(fertility[l] for l in SCORED_LANGUAGES) / len(SCORED_LANGUAGES)
        budget = raw * CONTEXT_FERTILITY_RATIO
        overages = {
            language: max(0.0, fertility.get(language, 0.0) - budget)
            for language in CONTEXT_LANGUAGES
        }
        penalty = sum(overages.values())
        score = sum(penalised[l] for l in SCORED_LANGUAGES) / len(SCORED_LANGUAGES)
        throughput, elapsed = _benchmark(tokenizer, rows, repeats=repeats)

        reconstruction = 1 - lossy / max(len(rows), 1)
        reconstruction_penalty = RECONSTRUCTION_PENALTY * (1 - reconstruction)

        report.update({
            "score": score + penalty + reconstruction_penalty,
            "fertility": fertility,
            "reconstruction_penalty": reconstruction_penalty,
            "unknown_rate": unknown_rate, "penalised": penalised,
            "lossy_rows": lossy, "reconstruction": reconstruction,
            "guardrail_budget": budget,
            "guardrail_overages": overages, "guardrail_penalty": penalty,
            "throughput": throughput, "elapsed_seconds": elapsed,
            "rows": len(rows),
        })

    report["valid"] = bool(tokenizer) and all(checks.values()) and not errors
    if verbose:
        _print_report(report)
    return report


def _print_guardrail(report: dict) -> None:
    """Explain the English and French guardrail in terms of headroom or cost."""
    budget = report["guardrail_budget"]
    charged = {
        language: value
        for language, value in report["guardrail_overages"].items()
        if value > 0.0
    }
    if charged:
        for language, overage in charged.items():
            print(f"  {LANGUAGE_NAMES[language]} is {overage:.3f} over the "
                  f"{budget:.3f} guardrail, adding {overage:.3f} to your score")
        return
    worst = max(CONTEXT_LANGUAGES, key=lambda l: report["fertility"].get(l, 0.0))
    value = report["fertility"].get(worst, 0.0)
    headroom = (budget - value) / budget * 100 if budget else 0.0
    print(f"  guardrail {budget:.3f}, highest is {LANGUAGE_NAMES[worst]} at "
          f"{value:.3f} ({headroom:.1f}% headroom)")
    if headroom < 5.0:
        print("  warning: this is close to the limit and the hidden test set "
              "differs slightly, so you may be charged there")


def _print_report(report: dict) -> None:
    """Print a submission report in the official checker layout."""
    print("AI Research Foundations Multilingual Tokenization Challenge")
    print("Submission checker")
    print()
    labels = {
        "loads": "Loading tokenizer",
        "file_size": "File size",
        "vocabulary": "Vocabulary",
        "encoding": "Encoding",
        "compatibility": "Compatibility",
    }
    for key, label in labels.items():
        if key not in report["checks"]:
            continue
        mark = "✓" if report["checks"][key] else "✗"
        detail = ""
        if key == "vocabulary" and report["vocab_size"] is not None:
            detail = f" {report['vocab_size']:,} / {MAX_VOCAB_SIZE:,}"
        print(f"{_leader(label)} {mark}{detail}")

    if "score" in report:
        print()
        print(f"Scores ({report['rows']:,} rows, lower is better)")
        print(f"{'language':<12}{'tokens/word':>13}{'[UNK] rate':>12}{'score':>9}")
        for language in LANGUAGES:
            if language not in report["fertility"]:
                continue
            marker = "*" if language in SCORED_LANGUAGES else " "
            print(f"{LANGUAGE_NAMES[language] + marker:<12}"
                  f"{report['fertility'][language]:>13.3f}"
                  f"{report['unknown_rate'][language]:>12.4f}"
                  f"{report['penalised'][language]:>9.3f}")
        print(f"{_leader('Guardrail penalty')} {report['guardrail_penalty']:.4f}")
        print(f"{_leader('Reconstruction penalty')} "
              f"{report['reconstruction_penalty']:.4f}")
        print(f"{_leader('SCORE')} {report['score']:.4f}")
        print("  * scored languages")
        _print_guardrail(report)
        print(f"  reconstruction {report['reconstruction'] * 100:.1f}%, "
              f"charged {RECONSTRUCTION_PENALTY:g} x the share not reconstructed")
        if report["reconstruction"] < 1.0:
            print(f"  {report['lossy_rows']:,} of {report['rows']:,} rows are not "
                  "reconstructed exactly, so some original text is lost")
        print()
        print("Local benchmark (informational only)")
        elapsed = report["elapsed_seconds"]
        readable = f"{elapsed:.2f} s" if elapsed >= 1 else f"{elapsed * 1000:.0f} ms"
        print(f"{_leader('Evaluation time')} {readable}")
        speed = report["throughput"] / 1e6
        print(f"{_leader('Throughput')} {speed:.1f}M characters/sec")

    print()
    print("READY FOR SUBMISSION ✓" if report["valid"] else "NOT READY FOR SUBMISSION")
    for error in report["errors"]:
        print(f"✗ {error}")


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "tokenizer.json"
    raise SystemExit(0 if profile_submission(target)["valid"] else 1)
