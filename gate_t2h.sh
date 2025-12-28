#!/bin/bash
# gate_t2h.sh - T+2h Gate: SKELETON
# RUN THIS OR KILL PROJECT

set -e

echo "=================================================="
echo "  AI FLIGHT RECORDER - T+2h Gate Check"
echo "=================================================="
echo ""

FAILURES=0

# Check required files
echo "Checking required files..."

if [ -f spec.md ]; then
    echo "  ✓ spec.md exists"
else
    echo "  ✗ FAIL: spec.md missing"
    FAILURES=$((FAILURES + 1))
fi

if [ -f ledger_schema.json ]; then
    echo "  ✓ ledger_schema.json exists"
else
    echo "  ✗ FAIL: ledger_schema.json missing"
    FAILURES=$((FAILURES + 1))
fi

if [ -f cli.py ]; then
    echo "  ✓ cli.py exists"
else
    echo "  ✗ FAIL: cli.py missing"
    FAILURES=$((FAILURES + 1))
fi

echo ""
echo "Running functional tests..."

# Test dual_hash
python3 -c "
from src.core import dual_hash
h = dual_hash(b'test')
assert ':' in h, 'dual_hash must return SHA256:BLAKE3 format'
print('  ✓ dual_hash returns dual format')
" || { echo "  ✗ FAIL: dual_hash test"; FAILURES=$((FAILURES + 1)); }

# Test Merkle tree
python3 -c "
from src.anchor import MerkleTree
t = MerkleTree()
t.add_leaf(b'a')
t.add_leaf(b'b')
root = t.get_root()
assert root, 'Merkle tree should have root'
print(f'  ✓ Merkle tree works (root: {root[:32]}...)')
" || { echo "  ✗ FAIL: Merkle tree test"; FAILURES=$((FAILURES + 1)); }

# Test 10-cycle smoke test
python3 -c "
from src.drone import run_cycle
state = {}
for _ in range(10):
    state, receipt = run_cycle(state)
print('  ✓ 10 decision cycles completed')
" || { echo "  ✗ FAIL: 10-cycle test"; FAILURES=$((FAILURES + 1)); }

# Test receipt emission
python3 cli.py --test 2>&1 | grep -q '"receipt_type"' && {
    echo "  ✓ cli.py emits valid receipts"
} || {
    echo "  ✗ FAIL: cli.py does not emit receipts"
    FAILURES=$((FAILURES + 1))
}

echo ""
echo "=================================================="

if [ $FAILURES -eq 0 ]; then
    echo "  ✓ PASS: T+2h gate - ALL CHECKS PASSED"
    echo "=================================================="
    exit 0
else
    echo "  ✗ FAIL: T+2h gate - $FAILURES checks failed"
    echo "=================================================="
    exit 1
fi
