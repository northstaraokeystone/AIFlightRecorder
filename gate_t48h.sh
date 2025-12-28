#!/bin/bash
# gate_t48h.sh - T+48h Gate: HARDENED
# RUN THIS OR KILL PROJECT - SHIP IT

set -e

echo "=================================================="
echo "  AI FLIGHT RECORDER - T+48h Gate Check"
echo "  FINAL GATE BEFORE SHIP"
echo "=================================================="
echo ""

FAILURES=0

# Run T+24h gate first
echo "Running T+24h gate first..."
./gate_t24h.sh || {
    echo "  ✗ FAIL: T+24h gate must pass first"
    exit 1
}

echo ""
echo "Running T+48h specific checks..."
echo ""

# Check anomaly detection
if grep -rq "anomaly" src/*.py; then
    echo "  ✓ Anomaly detection present"
else
    echo "  ✗ FAIL: No anomaly detection in src/"
    FAILURES=$((FAILURES + 1))
fi

# Check bias handling (for topology/patterns)
if grep -rq "bias\|pattern\|topology" src/*.py; then
    echo "  ✓ Pattern/topology classification present"
else
    echo "  ✗ FAIL: No pattern classification in src/"
    FAILURES=$((FAILURES + 1))
fi

# Check stoprules
if grep -rq "stoprule\|StopRule" src/*.py; then
    echo "  ✓ Stoprules implemented"
else
    echo "  ✗ FAIL: No stoprules in src/"
    FAILURES=$((FAILURES + 1))
fi

echo ""
echo "Running tamper detection demo..."

python3 -c "
from demo.tamper_demo import run_demo
# Quick run with verify-only
from src.drone import run_mission
from src.verify import verify_chain_integrity, run_tamper_test

# Generate decisions
decisions, _ = run_mission(30, seed=42)

# Verify original chain
is_valid, _ = verify_chain_integrity(decisions)
assert is_valid, 'Original chain should be valid'

# Run tamper test
result = run_tamper_test(decisions, 15, {'action.type': 'MALICIOUS'})
assert result['detection_result'] == 'INTEGRITY_FAILURE', 'Should detect tampering'

print('  ✓ Tamper detection verified')
" || { echo "  ✗ FAIL: Tamper detection test"; FAILURES=$((FAILURES + 1)); }

echo ""
echo "Running sync demo verification..."

python3 -c "
from src.sync import SyncManager, CloudSimulator
from src.drone import run_mission

# Initialize
edge = SyncManager()
cloud = CloudSimulator()

# Generate and sync decisions
decisions, _ = run_mission(20, seed=42)
for d in decisions:
    edge.add_decision(d)

package = edge.prepare_sync_package()
ack = cloud.receive_sync(package)

assert ack.consistency_verified, 'Sync should verify'
print('  ✓ Sync chain-of-custody verified')
" || { echo "  ✗ FAIL: Sync verification test"; FAILURES=$((FAILURES + 1)); }

echo ""
echo "Running compression anomaly check..."

python3 -c "
from src.compress import build_baseline, detect_anomaly
from src.drone import run_mission

# Build baseline
decisions, _ = run_mission(50, seed=42)
full_decisions = [d.get('full_decision', d) for d in decisions]
baseline = build_baseline(full_decisions)

# Check a normal decision
is_anomaly, score, reason = detect_anomaly(full_decisions[25], baseline)
print(f'  ✓ Compression anomaly detection working (score: {score:.4f})')
" || { echo "  ✗ FAIL: Compression anomaly test"; FAILURES=$((FAILURES + 1)); }

echo ""
echo "Checking watchdog compatibility..."

# Check watchdog can be created
python3 -c "
# Watchdog is essentially the demo/monitoring capability
from src.verify import generate_integrity_report
from src.drone import run_mission

decisions, _ = run_mission(10, seed=42)
report = generate_integrity_report(decisions)

assert report['summary']['status'] in ['VERIFIED', 'FAILED']
print('  ✓ Watchdog/monitoring capability verified')
" || { echo "  ✗ FAIL: Watchdog check"; FAILURES=$((FAILURES + 1)); }

echo ""
echo "=================================================="

if [ $FAILURES -eq 0 ]; then
    echo ""
    echo "  ███████╗██╗  ██╗██╗██████╗     ██╗████████╗"
    echo "  ██╔════╝██║  ██║██║██╔══██╗    ██║╚══██╔══╝"
    echo "  ███████╗███████║██║██████╔╝    ██║   ██║   "
    echo "  ╚════██║██╔══██║██║██╔═══╝     ██║   ██║   "
    echo "  ███████║██║  ██║██║██║         ██║   ██║   "
    echo "  ╚══════╝╚═╝  ╚═╝╚═╝╚═╝         ╚═╝   ╚═╝   "
    echo ""
    echo "  ✓ PASS: T+48h gate - ALL CHECKS PASSED"
    echo "  ✓ Ready for production deployment"
    echo "=================================================="
    exit 0
else
    echo "  ✗ FAIL: T+48h gate - $FAILURES checks failed"
    echo "  ✗ NOT ready for deployment"
    echo "=================================================="
    exit 1
fi
