# Maick Dane Nkou — lossless BPE, weights-b4

## Submitted recipe

- **weights-b4**: BPE with 10,000 vocabulary entries, minimum frequency 5.
- Lossless `space_word` boundaries: isolated `Split(Regex(r" ?\S+|\s+"))`,
  followed by `ByteLevel(add_prefix_space=False, use_regex=False)`.
- Full 256-symbol byte alphabet and matching ByteLevel decoder.
- No normalizer, special tokens or post-processor.
- Official train split only: 240,000 original rows, balanced round-robin;
  **ha/sw/yo/am x4, en/fr x1** (720,000 weighted rows).
- Dataset: `Similoluwa/african-multilingual-tokenizer-challenge` @ `v1.0.0`.
- `tokenizers==0.22.1`; no pretrained tokenizer, external corpus, downloaded
  vocabulary or merge table.

## Recorded official-checker validation results

These measurements were produced in the participant's Colab session on all
**24,000 official validation rows** (4,000 per language), using the pinned
`profile_submission` helper. They are not hidden-test or leaderboard scores.
The uploaded tokenizer's SHA-256 matches the b4 file evaluated in those runs.
The final repository checks verify this identity; they do not claim a new
full-validation run in the repository environment.

| Language | Tokens/word | UNK rate |
| --- | ---: | ---: |
| English | 2.063796 | 0.000000 |
| French | 2.190751 | 0.000000 |
| Hausa | 1.662510 | 0.000000 |
| Swahili | 1.860250 | 0.000000 |
| Yoruba | 1.866333 | 0.000000 |
| Amharic | 2.369574 | 0.000000 |

Per-language values above are rounded to six decimal places.

- **Full validation score: 1.939666974** (rounded to nine decimal places).
- Base score: 1.939666974; guardrail penalty: 0; reconstruction penalty: 0.
- Official reconstruction: 100%; the notebook also passed stricter exact
  reconstruction checks on every validation string.
- EN/FR headroom: **1.787199439%** (rounded to nine decimal places).
- Tokenizer SHA-256:
  `b2f9a461ce1b0e8bff07c00d982614f49d555464b8fb1efa46e66b08b67013e4`.
- Official checker commit: `75578f2400c39b1f8e31ce7e7104b37fbc470d11`.
- Official checker SHA-256:
  `1727de34136097eb48addabf90501589bdfefa31c20e201bab53c38f2f7c9688`.

The full score includes the mean target-language fertility/UNK component and
both official penalties. The former participant-imposed 5% headroom policy is
not an official rule and is not enforced here. The small remaining margin means
that penalties are possible on different test data; no hidden-test score or rank
is guaranteed.

## Change from the previous submission

This file replaces `weights-yo-am4`, previously measured at 1.9631891569 on the
same public validation split, with SHA-256
`1519895eace8680d2752b333f5efd82f80ade21bcd6210704ec17de55dafe035`.
The validation score decreases by approximately **1.20%**, with less EN/FR margin.
The participant selected b4 rather than the lowest-scoring optimization candidate
because that alternative had only about 0.1% margin.

## Reproduce

Run `notebook.ipynb` end-to-end in Colab on CPU. The default
`RECIPE = "weights-b4"` trains from scratch on the complete official train split,
then evaluates the saved tokenizer on the complete official validation split.
No uploaded tokenizer or diagnostic archive is required.

After complete validation, zero UNK, zero official penalties and strict
reconstruction checks, Colab automatically downloads `tokenizer.json`,
`metadata.yml` and `README.md`. Allow multiple downloads if prompted.
A failed check blocks export, not the display of the official score.
The 5% headroom policy is informational, not an export gate.

Exports are written to
`artifacts/submission-weights-b4/export/maick-dane-nkou/`, not over the repository
submission. The generated README records the new score, file SHA-256, data
fingerprints and training time. No report-upload step or review archive is used.
Downloading files does not automatically submit them to GitHub.

BPE merge ties can vary between training runs. Always use the new file's measured
score rather than assuming that retraining will reproduce the historical bytes.
The optional `RECIPE = "weights-yo-am4"` reproduces the **previous** recipe, not
the tokenizer currently submitted here.
