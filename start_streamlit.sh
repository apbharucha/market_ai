#!/bin/bash

# Octavian Market AI - Clean Streamlit Startup Script
# This ensures all changes are loaded fresh

echo "========================================================================"
echo "OCTAVIAN MARKET AI - CLEAN START"
echo "========================================================================"

# 1. Kill any existing Streamlit processes
echo "1. Stopping any running Streamlit processes..."
pkill -9 -f "streamlit run" 2>/dev/null
sleep 2
echo "   ✓ Stopped"

# 2. Clear all caches
echo "2. Clearing Python and Streamlit caches..."
rm -rf __pycache__ .streamlit/cache trading_system/__pycache__ 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
echo "   ✓ Caches cleared"

# 3. Verify changes are present
echo "3. Verifying implementation..."
if grep -q "Breaking Trades" paper_trading_ui.py; then
    echo "   ✓ Breaking Trades tab found"
else
    echo "   ✗ WARNING: Breaking Trades tab not found"
fi

if grep -q "Performance Dashboard" paper_trading_ui.py; then
    echo "   ✓ Performance Dashboard found"
else
    echo "   ✗ WARNING: Performance Dashboard not found"
fi

# 4. Check modules load correctly
echo "4. Testing module imports..."
python3 -c "
from paper_trading_engine import PaperTradingEngine
from trading_system.risk_manager import AdvancedRiskManager
from trading_system.simulation_grader import SimulationGrader
from breaking_trades_generator import BreakingTradesGenerator
print('   ✓ All modules import successfully')
" 2>&1 | grep -v "WARNING"

echo ""
echo "========================================================================"
echo "STARTING STREAMLIT WITH FRESH ENVIRONMENT"
echo "========================================================================"
echo ""
echo "Changes included:"
echo "  • Breaking Trades tab (high-confidence setups)"
echo "  • Performance Dashboard with grading"
echo "  • Risk Management system"
echo "  • Enhanced PnL tracking"
echo ""
echo "Navigate to: Paper Trading"
echo "You should see 5 tabs: Breaking Trades | Equity | Options | Positions | History"
echo ""
echo "========================================================================"
echo ""

# 5. Start Streamlit with clean state
streamlit run main.py --server.headless true
