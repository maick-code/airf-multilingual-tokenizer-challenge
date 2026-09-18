# Maick Dane Nkou — lossless BPE, raw-sw75-ha425-yo55-am55-fr128125

## Proposed recipe

- **raw-sw75-ha425-yo55-am55-fr128125**: BPE with 10,000 vocabulary entries and
  minimum frequency 5.
- **Raw ByteLevel pieces**: the pre-tokenizer is a single
  `ByteLevel(add_prefix_space=False, use_regex=False)` with **no whitespace
  split**, so BPE merges may cross word boundaries. This is the only
  architectural change versus the previous `fr125-yo5-am5` submission; with
  identical weights it alone improved the full-validation score from 1.938409
  to 1.930205 before any re-tuning.
- Language weights in the order en, fr, ha, sw, yo, am:
  **x1, x1.28125, x4.25, x7.5, x5.5, x5.5** (units 32, 41, 136, 240, 176, 176
  over denominator 32, applied with the deterministic cumulative train-only
  schedule, so every one of the 240,000 train rows appears at least once).
- Full 256-symbol byte alphabet
  (`initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet())`) and matching
  ByteLevel decoder.
- No normalizer, special tokens, post-processor or UNK token.
- Official train split only: 240,000 original rows, 1,001,250 weighted rows.
- Dataset: `Similoluwa/african-multilingual-tokenizer-challenge` @ `v1.0.0`.
- `tokenizers==0.22.1`; no pretrained tokenizer, external corpus, downloaded
  vocabulary or merge table. Three independent training runs produced the
  identical bytes (see fingerprint below).

## Measured public-validation results

Measured with the repository's own scoring code
(`competition.evaluation.evaluate_submission` and the pinned
`profile_submission` helper) on all **24,000 official validation rows**
(4,000 per language). These are validation measurements, not private-test or
final leaderboard scores.

| Language | Tokens | Words | Tokens/word | UNK rate |
| --- | ---: | ---: | ---: | ---: |
| English | 184,137 | 88,469 | 2.081373 | 0.000000 |
| French | 196,994 | 90,652 | 2.173079 | 0.000000 |
| Hausa* | 151,067 | 89,819 | 1.681905 | 0.000000 |
| Swahili* | 123,154 | 69,832 | 1.763575 | 0.000000 |
| Yoruba* | 159,815 | 85,758 | 1.863558 | 0.000000 |
| Amharic* | 188,443 | 79,413 | 2.372949 | 0.000000 |

\* scored languages.

- **Full validation score: 1.9204967724666027** (previous submission
  fr125-yo5-am5: 1.938409033803239; improvement 0.017912).
- Base score: 1.9204967724666027; guardrail penalty: 0; reconstruction
  penalty: 0.
- Official reconstruction: 100%; the notebook additionally passes a stricter
  exact-match audit of `decode(encode(text)) == text` on every validation row
  (both `skip_special_tokens` modes) and on adversarial regression strings.
- Guardrail: budget = 1.15 x scored mean = 2.208571; English 2.081373 and
  French 2.173079 are both under it. French headroom is 1.6%, comparable to
  previous submissions; small hidden-test drifts could in principle incur a
  small guardrail charge there, nothing is guaranteed.
- Runtime: evaluation throughput ~5.0-5.3M characters/second on the research
  machine, i.e. ~0.95x the character-level baseline time — well inside the 5x
  limit (the official leaderboard protocol check with the shipped
  character-level baseline passed with no failure).

## Fingerprints

- Tokenizer SHA-256:
  `4043a0499c1e4578bbc636a91f580abfe77dbe0ac52af4dbb0335743e973f5ed`
  (identical across three independent training runs).
- Official train serialization used for training (round-robin
  en/fr/ha/sw/yo/am JSONL): SHA-256
  `85140695f33ff87ead838dbac449fbc7f13aec72dde4129556d7538ba17cbc4f`.
- Official CSV exports mirrored from
  `Similoluwa/african-multilingual-tokenizer-challenge` @ `v1.0.0`:
  train SHA-256
  `bf807cb78e58a6dfb3ba50652013d54fb1d2f9f5e6eec4c98f22d5989364efd7`,
  validation SHA-256
  `d010df416334c0f9086d0831e7aa19c2a7d233b08032e2a7d3118f512d3c0f00`.

## Research summary (what was tried)

All candidates below were trained only on the official train split and
evaluated on the full 24,000-row official validation split.

1. **Baseline audit.** The previous fr125-yo5-am5 tokenizer was reproduced
   byte-for-byte locally (SHA-256
   `6f387416333fdfff5d45135313b00d5ef5ffd580b4c15ad851cd01f13b81d0e5`) and
   re-measured at exactly 1.938409033803239, confirming the baseline.
2. **Weight sweep (word-boundary architecture).** Pushing Swahili weight
   (x4 -> x7.5) and adjusting Hausa/Yoruba/Amharic improved the score to
   1.929248 (sw75-ha4), but squeezed the French guardrail margin.
3. **min_frequency sweep (3/5/7/8).** No effect at these weights: the top
   10,000 merges all sit well above the threshold. Not a lever here.
4. **Structural variants.** WordPiece failed (the library requires an UNK
   token at encode time, which is incompatible with zero-UNK lossless
   submission). Punctuation-splitting pre-regexes hurt badly (2.031845).
5. **Raw pieces (the winning change).** Removing the whitespace split so
   merges may cross word boundaries improved every configuration tested:
   fr125 weights 1.938409 -> 1.930205; retuned weights
   (en x1, fr x1.28125, ha x4.25, sw x7.5, yo x5.5, am x5.5) ->
   **1.920497**. The gain comes mostly from high-frequency cross-word
   fragments in the Latin-script languages; it also lowers French fertility,
   which restores guardrail margin.
6. **Unigram (explored, rejected for non-reproducibility).** A Unigram model
   with the same raw-piece architecture scored **1.879618** on the same split
   (ha 1.6177, sw 1.6630, yo 1.8104, am 2.4275; zero UNK, 100%
   reconstruction). However, two independent runs of the identical recipe
   produced *different* vocabularies (9,233 of 10,000 pieces differ) even
   though both score identically to six decimals: the EM pruning is not
   byte-deterministic. Since the submission contract requires a notebook that
   reproduces the submitted bytes exactly, Unigram was rejected. Weight
   retunes of it also either sacrificed the Amharic lead (am 2.4275) or
   pushed the French guardrail margin below 1%.

Selected candidates on the full validation split:

| Candidate | Score | ha | sw | yo | am | fr headroom |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fr125-yo5-am5 (previous submission) | 1.938409 | 1.6862 | 1.8814 | 1.8537 | 2.3323 | 2.1% |
| sw75-ha4 (word boundaries) | 1.929248 | 1.6958 | 1.7797 | 1.8671 | 2.3744 | 0.3% |
| raw-fr125 (raw pieces, old weights) | 1.930205 | 1.6656 | 1.8714 | 1.8456 | 2.3381 | 4.0% |
| raw-sw75-ha4 | 1.925698 | 1.6889 | 1.7506 | 1.8754 | 2.3879 | 2.2% |
| **raw-sw75-ha425-yo55-am55-fr128125 (this submission)** | **1.920497** | 1.6819 | 1.7636 | 1.8636 | 2.3729 | 1.6% |

The submitted candidate improves Hausa, Swahili, English and French versus
the previous submission, keeps Yoruba within 0.01 and Amharic within 0.041
of it, and stays ahead of the current leaderboard's best Yoruba and Amharic
figures. The alternative `raw-sw75-ha425-yo55` scored 1.921879 with 1.9%
French headroom but pushes Amharic to 2.4011; the submitted variant was
preferred for its more balanced profile across all four scored languages.

The submitted BPE was reproduced byte-for-byte in four independent settings:
three training runs in the research sandbox and one run on a clean CI runner
from a fresh Hub download of the official dataset (see
`.github/scripts/remote_train.py` on the research branch), all yielding the
same SHA-256.

## Reproduce

Run `notebook.ipynb` end-to-end on CPU (Google Colab or a clone). It loads
the official train split of
`Similoluwa/african-multilingual-tokenizer-challenge` @ `v1.0.0`, trains the
tokenizer from scratch with the exact recipe above, and asserts that the
resulting `tokenizer.json` has SHA-256
`4043a0499c1e4578bbc636a91f580abfe77dbe0ac52af4dbb0335743e973f5ed` before
evaluating it on the full official validation split. An offline fallback is
included: if the two official CSV exports are already present locally, the
notebook uses them after checking their SHA-256 fingerprints.
