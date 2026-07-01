#!/bin/bash
# Run Kimi-K2.6 (thinking) multi-agent eval sequentially under 3 orchestrator prompt styles.
# Each is a full 380-run multi eval; results go to generated_new_200_<style>/multi__Kimi-K2.6__...
cd /shared/dev/jwanyang/multi-agent-world-model
source export.sh >/dev/null 2>&1
for st in neutral delegate solo; do
  echo "===== START Kimi multi style=$st  $(date) ====="
  python eval_main.py --init "eval/config/multi3/$st/kimi26.yml" --parallel 8
  echo "===== DONE  Kimi multi style=$st  $(date) ====="
done
echo "===== ALL 3 STYLES DONE  $(date) ====="
