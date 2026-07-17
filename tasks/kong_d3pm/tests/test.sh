#!/bin/bash
# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

set -e
cd /app/tasks/kong_d3pm

mkdir -p /logs/verifier

# ---------------------------------------------------------------------------
# Strategy: independently evaluate the agent's best checkpoint via score.py.
# Fall back to self-reported summary.tsv only if score.py fails.
# ---------------------------------------------------------------------------

SCORE_OK=0

if [ -f outputs/best_model.pt ]; then
    echo "Found outputs/best_model.pt -- running independent evaluation ..."
    SCORE_JSON=$(python score.py --checkpoint outputs/best_model.pt --horizons 1 10 20 2>/tmp/score_stderr) && SCORE_OK=1 || true

    if [ "$SCORE_OK" = "1" ] && [ -n "$SCORE_JSON" ]; then
        # Weighted multi-horizon score: 0.1*h1 + 0.2*h10 + 0.7*h20
        WEIGHTED_SCORE=$(echo "$SCORE_JSON" | python3 -c "
import sys, json
d = json.load(sys.stdin)
h1 = d.get('h1', {}).get('composite', 0.0)
h10 = d.get('h10', {}).get('composite', 0.0)
h20 = d.get('h20', {}).get('composite', 0.0)
score = 0.1 * h1 + 0.2 * h10 + 0.7 * h20
print(round(score, 6))
" 2>/dev/null) || WEIGHTED_SCORE=""

        if [ -n "$WEIGHTED_SCORE" ] && [ "$WEIGHTED_SCORE" != "None" ]; then
            echo "$WEIGHTED_SCORE" > /logs/verifier/reward.txt
            echo "$SCORE_JSON" > /logs/verifier/rewards.json
            echo "Weighted score (0.1*h1 + 0.2*h10 + 0.7*h20): $WEIGHTED_SCORE"
            exit 0
        else
            echo "WARNING: could not parse weighted score from score.py output"
            SCORE_OK=0
        fi
    else
        echo "WARNING: score.py failed or produced no output"
        [ -f /tmp/score_stderr ] && cat /tmp/score_stderr
        SCORE_OK=0
    fi
else
    echo "No outputs/best_model.pt found -- skipping independent eval"
fi

# ---------------------------------------------------------------------------
# Fallback: read self-reported composite from summary.tsv
# ---------------------------------------------------------------------------
echo "Falling back to summary.tsv ..."

if [ ! -f summary.tsv ] || [ $(wc -l < summary.tsv) -le 1 ]; then
    echo "0.0" > /logs/verifier/reward.txt
    echo '{"composite": 0.0, "source": "no_experiments"}' > /logs/verifier/rewards.json
    echo "No experiments found in summary.tsv"
    exit 0
fi

BEST_COMPOSITE=$(tail -n +2 summary.tsv | awk -F'\t' '{print $4}' | grep -v '^$' | sort -rn | head -1)

if [ -z "$BEST_COMPOSITE" ]; then
    echo "0.0" > /logs/verifier/reward.txt
    echo '{"composite": 0.0, "source": "no_composite_in_tsv"}' > /logs/verifier/rewards.json
    echo "No composite score found"
    exit 0
fi

NUM_EXPERIMENTS=$(tail -n +2 summary.tsv | wc -l)

echo "$BEST_COMPOSITE" > /logs/verifier/reward.txt
cat > /logs/verifier/rewards.json << ENDJSON
{
    "composite": $BEST_COMPOSITE,
    "num_experiments": $NUM_EXPERIMENTS,
    "source": "summary_tsv_fallback"
}
ENDJSON

echo "Best composite (self-reported): $BEST_COMPOSITE (from $NUM_EXPERIMENTS experiments)"
