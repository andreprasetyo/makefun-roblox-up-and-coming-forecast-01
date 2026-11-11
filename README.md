# Roblox Up-and-Coming forecast — diffusion-with-drift

A pure-code statistical forecaster (Node 22, standard library only, no
network, no model calls) for make.fun's `roblox-up-and-coming-forecast-01`.

## How it works

For each game on the chart, its rank at horizon h hours after the history
ends is modelled as Normal(current rank + drift(h), sigma(h)):

- **drift** — the rank band's mean 24 h rank change, estimated from every
  pair of reads ~24 h apart in the history, plus the game's own recent rank
  velocity (shrunk by 0.10, averaged over two consecutive ~24 h steps) and
  its recent log-playing growth (coefficient −3; playing growth predicts
  rank climbs, forward R² 0.069 alone, 0.075 with rank velocity);
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

Scored with an exact replica of the sealed scorer, on the public archive's
reads (the practice window's 10 reads were rebuilt from the archive and
verified against the kit's history lines; the server grade reproduces the
practice score to 6 decimals):

| set | this model | previous version | persistence baseline |
| --- | --- | --- | --- |
| practice window (10 reads) | **−31.80** | −33.71 | −43.51 |
| walk-forward, spread folds (6) | **−36.94** | −39.25 | −49.68 |
| walk-forward, late folds (6) | **−31.86** | −34.22 | — |

Tuned offline (grid search + walk-forward guard); the recommended config
improves on the previous version on 10 of 11 walk-forward folds and clears
the top-10 practice bar (−32.0).

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Deterministic: two runs on the same history write byte-identical output.
