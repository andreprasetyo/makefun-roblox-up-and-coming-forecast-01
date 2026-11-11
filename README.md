# Roblox Up-and-Coming forecast — diffusion-with-drift

A pure-code statistical forecaster (Node 22, standard library only, no
network, no model calls) for make.fun's `roblox-up-and-coming-forecast-01`.

## How it works

For each game on the chart, its rank at horizon h hours after the history
ends is modelled as Normal(current rank + drift(h), sigma(h)):

- **drift** — the rank band's mean 24 h rank change, estimated from every
  pair of reads ~24 h apart in the history, plus the game's own recent rank
  velocity (shrunk by 0.25; momentum persists, corr ~0.24);
- **sigma** — the band's standard deviation of 24 h rank changes, scaled by
  sqrt(h / 24) and by 0.6 (the horizon average already spreads the forecast).

Per-place probabilities are the time-average over horizons 0–72 h, so early
hours stay close to the chart as it stands and later hours spread out as the
chart churns. Each place leaves 15% of its mass unnamed — the scorer shares
it over the games on a read's chart the forecast does not name there, which
covers brand-new games (in the practice window a game never seen in the
history reached #1 in 3 of 10 reads). When a read carries the games'
details, a game's `created` date is used too: the chart only lists games
published in the last 28 days, so a game gets no mass at horizons past its
28th day.

## Backtest

Scored with an exact replica of the sealed scorer
(`grader/forecast/ranked.ts`), on the public archive's reads (the practice
window's 10 reads were rebuilt from the archive and verified byte-for-byte
against the kit's history lines):

| fold (cutoff) | this model | persistence baseline |
| --- | --- | --- |
| 2026-11-02 | −35.59 | −46.88 |
| 2026-11-03 | −39.75 | −50.89 |
| 2026-11-04 | −40.27 | −48.64 |
| 2026-11-05 | −43.00 | −55.85 |
| 2026-11-06 | −39.09 | −52.45 |
| 2026-11-07 (practice) | −33.71 | −43.51 |
| **mean** | **−38.57** | **−49.68** |

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Deterministic: two runs on the same history write byte-identical output.
