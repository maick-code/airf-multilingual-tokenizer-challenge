# Maick Dane Nkou — lossless BPE, fr125-yo5-am5

## Proposed recipe

- **fr125-yo5-am5**: BPE with 10,000 vocabulary entries and minimum frequency 5.
- Language weights in the order en, fr, ha, sw, yo, am: **x1, x1.25, x4, x4, x5, x5**.
- The fractional French weight uses the deterministic cumulative train-only schedule from the research notebook.
- Lossless `space_word` boundaries: isolated `Split(Regex(r" ?\S+|\s+"))`,
  followed by `ByteLevel(add_prefix_space=False, use_regex=False)`.
- Full 256-symbol byte alphabet and matching ByteLevel decoder.
- No normalizer, special tokens or post-processor.
- Official train split only: 240,000 original rows and 810,000 weighted rows.
- Dataset: `Similoluwa/african-multilingual-tokenizer-challenge` @ `v1.0.0`.
- `tokenizers==0.22.1`; no pretrained tokenizer, external corpus, downloaded
  vocabulary or merge table.

## Measured public-validation results

The following measurements were obtained twice in independent Colab training
runs on all **24,000 official validation rows** (4,000 per language). They are
validation measurements, not private-test or final leaderboard scores.

Both runs produced the same tokenizer bytes and the same metrics. The current
b4 artifact remains the reference until a participation PR is accepted.

| Language | Tokens/word | UNK rate |
| --- | ---: | ---: |
| English | 2.075710 | 0.000000 |
| French | 2.183438 | 0.000000 |
| Hausa | 1.686191 | 0.000000 |
| Swahili | 1.881415 | 0.000000 |
| Yoruba | 1.853716 | 0.000000 |
| Amharic | 2.332313 | 0.000000 |

Per-language values above are rounded to six decimal places.

- **Full validation score: 1.9384090338032391**.
- Base score: 1.9384090338032391.
- Guardrail penalty: 0.
- Reconstruction penalty: 0.
- Official reconstruction: 100%; `lossy_rows = 0`.
- Strict reconstruction: 0 failures on all 24,000 validation strings.
- EN/FR headroom: **2.0515536681751971%**.
- Official checker score: 1.9384090338032391.
- Official checker rows: 24,000.
- Tokenizer SHA-256:
  `6f387416333fdfff5d45135313b00d5ef5ffd580b4c15ad851cd01f13b81d0e5`.
- Train serialization SHA-256:
  `85140695f33ff87ead838dbac449fbc7f13aec72dde4129556d7538ba17cbc4f`.
- Official checker commit: `75578f2400c39b1f8e31ce7e7104b37fbc470d11`.
- Official checker SHA-256:
  `1727de34136097eb48addabf90501589bdfefa31c20e201bab53c38f2f7c9688`.

The score includes the target-language fertility/UNK component and both
official penalties. The former participant-imposed 5% headroom policy is not
an official rule. No private-test score, final rank or hidden-test improvement
is guaranteed by these validation measurements.

## Comparison with the b4 reference

The b4 reference was measured on the same public validation split at
`1.9396669738781722`, with EN/FR headroom `1.7871994392793225%` and SHA-256
`b2f9a461ce1b0e8bff07c00d982614f49d555464b8fb1efa46e66b08b67013e4`.

The proposed tokenizer therefore has a lower measured validation score and a
higher measured EN/FR margin, while preserving zero UNK and exact
reconstruction. These are public-validation comparisons only; the official
leaderboard evaluates different public/private test texts.

## Reproduction and submission status

The tokenizer was trained from scratch on CPU with the official train split.
The two independent runs used the pinned checker and `tokenizers==0.22.1`.
BPE merge ties may vary between environments, so any future artifact must be
re-evaluated from its own bytes.

This directory is being prepared on the Arena research branch. No review
reports, corpus files, caches or research artifacts are included in the
submission directory, and no automatic private-test claim is made.
