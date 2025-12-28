#!/bin/bash
# gate_t24h.sh - T+24h Gate: MVP
# RUN THIS OR KILL PROJECT

set -e

echo "=================================================="
echo "  AI FLIGHT RECORDER - T+24h Gate Check"
echo "=================================================="
echo ""

FAILURES=0

# Run T+2h gate first
echo "Running T+2h gate first..."
./gate_t2h.sh || {
    echo "  ✗ FAIL: T+2h gate must pass first"
    exit 1
}

echo ""
echo "Running T+24h specific checks..."
echo ""

# Run tests
echo "Running pytest..."
if python3 -m pytest tests/ -q --tb=short 2>&1; then
    echo "  ✓ All tests pass"
else
    echo "  ✗ FAIL: Tests failed"
    FAILURES=$((FAILURES + 1))
fi

echo ""
echo "Checking code patterns..."

# Check emit_receipt in src files
if grep -rq "emit_receipt" src/*.py; then
    echo "  ✓ emit_receipt found in src/"
else
    echo "  ✗ FAIL: No emit_receipt in src/"
    FAILURES=$((FAILURES + 1))
fi

# Check assertions in tests
if grep -rq "assert" tests/*.py; then
    echo "  ✓ Assertions found in tests/"
else
    echo "  ✗ FAIL: No assertions in tests/"
    FAILURES=$((FAILURES + 1))
fi

# Check dual_hash usage
if grep -rq "dual_hash" src/*.py; then
    echo "  ✓ dual_hash used in src/"
else
    echo "  ✗ FAIL: dual_hash not used in src/"
    FAILURES=$((FAILURES + 1))
fi

echo ""
echo "Running validation scenarios..."

python3 -c "
from sim.sim import run_all_scenarios
from sim.scenarios import QUICK_SCENARIOS

results = run_all_scenarios(QUICK_SCENARIOS)

if results['all_passed']:
    print('  ✓ Quick scenarios pass')
else:
    print('  ✗ FAIL: Quick scenarios failed')
    exit(1)
" || { FAILURES=$((FAILURES + 1)); }

echo ""
echo "=================================================="

if [ $FAILURES -eq 0 ]; then
    echo "  ✓ PASS: T+24h gate - ALL CHECKS PASSED"
    echo "=================================================="
    exit 0
else
    echo "  ✗ FAIL: T+24h gate - $FAILURES checks failed"
    echo "=================================================="
    exit 1
fi
