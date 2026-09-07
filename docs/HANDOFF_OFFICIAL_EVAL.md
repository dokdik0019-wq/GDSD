# GDSD official pr_eval (uint8) — HANDOFF for resume after Mac closed

Status: 2026-09-07 ~20:15 ICT — job RUNNING on box sc7308 (systemd user unit).

## What is running
- Unit: `gdsd-pr-u8` (systemctl --user on sc7308) — run_official_pr_u8.sh
- Official BSDS pr_evaluation: uint8 PNG soft maps, 99 global linspace
  thresholds over [0,1], interp AP (mirrors official MATLAB collect_eval_bdry.m)
- Methods: gdsd_v1fix, gdsd2, gdsd2_s28, canny, haralick x val+test (10 runs)
- 10 workers; ~15-30 min val, ~30-60 min test per method (faster than
  earlier 4-worker runs)

## Progress so far (official pr_eval, uint8 PNG, 99 thr)
| run | ODS | OIS | AP | status |
|---|---|---|---|---|
| gdsd_v1fix val | 0.5565 | 0.6069 | 0.3069 | DONE |
| gdsd_v1fix test | 0.5743 | 0.6014 | 0.3142 | DONE |
| gdsd2 val | — | | | RUNNING |
| gdsd2 test | — | | | queued |
| gdsd2_s28 val/test | — | | | queued |
| canny val/test | — | | | queued |
| haralick val/test | — | | | queued |

## Check progress
```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 -i ~/.ssh/id_ed25519 sc7308@sc7308-nitro-n50-120 \
  'systemctl --user is-active gdsd-pr-u8; cat ~/gdsd-bsds/official_pr_u8_results/run.log'
# per-run results land in ~/gdsd-bsds/official_pr_u8_results/<method>_<split>.log
```

## When ALL_OFFICIAL_PR_U8_DONE appears
1. Read all 10 logs in ~/gdsd-bsds/official_pr_u8_results/*.log
2. Replace the percentile-protocol numbers in:
   - ~/Desktop/KhonKaen2026/GDSD/README.md (BSDS500 results tables)
   - ~/Desktop/KhonKaen2026/GDSD/docs/BENCHMARK.md (result tables)
   with the official pr_eval numbers (ODS/OIS/AP per method val+test).
3. git add + commit + push (repo dokdik0019-wq/GDSD, local at
   ~/Desktop/KhonKaen2026/GDSD, branch main).
4. Report: v2 (gm@ZC) vs v1 (|R|@ZC) vs Canny vs Haralick, official
   protocol, ODS/OIS/AP tables. Note AP is lower than the old percentile
   AP because official AP = area under real PR curve (classical detectors
   have low AP by nature; literature Canny ODS ~0.58 matches).

## Repo staging (as of close)
Staged in git (not yet committed):
- benchmark/run_official_pr.py (official pipeline, 10-worker)
- benchmark/make_soft_pngs_u8.py (uint8 PNG = official imread/255 conv)
- benchmark/README.md, docs/BENCHMARK.md (official protocol description)
- removed: benchmark/make_soft_pngs_official.py (wrong global-max approach)
Legacy bsds_official_eval.py (percentile variant) kept as reference.

## Key lesson this session
Official BSDS MATLAB loads soft maps as PNG and does /255, then thresholds
on linspace(1/(N+1), 1-1/(N+1), N) over [0,1]. Correct encoding for
classical detectors = uint8 PNG (255*raw, clip) — NOT global-max uint16.
Per-image max then = 1.0 after /255 automatically. Haralick soft maps have
low natural scale (<0.55 after /255) — that is a property of the method.

## Watcher on Mac (dies with Mac)
proc_50e57f8d281d polls every 120s and prints logs when done. It will not
survive a Mac shutdown; resume manually per above after reopening.
