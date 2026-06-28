#!/bin/bash
# Run GLM-5.2 multi-agent eval sequentially under 3 orchestrator prompt styles.
# Each is a full 380-run multi eval; results go to generated_new_200_<style>.
cd /shared/dev/jwanyang/multi-agent-world-model
source export.sh >/dev/null 2>&1
for st in neutral delegate solo; do
  echo "===== START GLM multi style=$st  $(date) ====="
  python eval_main.py --init "eval/config/multi3/$st/glm52.yml" --parallel 4
  echo "===== DONE  GLM multi style=$st  $(date) ====="
done
echo "===== ALL 3 STYLES DONE  $(date) ====="
