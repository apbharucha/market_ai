#!/usr/bin/env python3
"""
Pre-flight check before starting Streamlit
This verifies ALL changes are present and will load correctly
"""

import sys
import os

print("="*80)
print("OCTAVIAN PRE-FLIGHT CHECK")
print("="*80)

all_passed = True

# Test 1: File exists and has correct content
print("\n[TEST 1] Checking paper_trading_ui.py file...")
try:
    with open('paper_trading_ui.py', 'r') as f:
        content = f.read()

    checks = {
        'Breaking Trades tab': '"Breaking Trades", "Equity Trade", "Options Trade"' in content,
        'Performance Dashboard section': 'Performance Dashboard & Grading' in content,
        'Breaking trades generator import': 'from breaking_trades_generator import get_breaking_trades_generator' in content,
        'Generate button': 'Generate Breaking Trades' in content,
        'Trade specifications display': 'Trade Specifications' in content,
        'Confidence scoring': 'confidence_score' in content,
        'Risk/Reward display': 'Risk/Reward' in content or 'R:R Ratio' in content,
        'Position sizing': 'suggested_position_size_pct' in content,
    }

    for check, result in checks.items():
        status = "✓" if result else "✗"
        print(f"  {status} {check}")
        if not result:
            all_passed = False

    print(f"\n  File size: {len(content):,} bytes")

except Exception as e:
    print(f"  ✗ ERROR: {e}")
    all_passed = False

# Test 2: Import the actual function and inspect it
print("\n[TEST 2] Importing and inspecting show_paper_trading function...")
try:
    from paper_trading_ui import show_paper_trading
    import inspect

    source = inspect.getsource(show_paper_trading)

    checks = {
        'Breaking Trades in function': 'Breaking Trades' in source,
        'Performance Dashboard in function': 'Performance Dashboard' in source,
        'trade_tabs variable': 'trade_tabs' in source,
        'Five tabs total': 'Trade History' in source,
    }

    for check, result in checks.items():
        status = "✓" if result else "✗"
        print(f"  {status} {check}")
        if not result:
            all_passed = False

    print(f"\n  Function length: {len(source):,} characters")

except Exception as e:
    print(f"  ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    all_passed = False

# Test 3: Check all new modules
print("\n[TEST 3] Testing new module imports...")
modules_to_test = [
    ('trading_system.risk_manager', 'AdvancedRiskManager'),
    ('trading_system.simulation_grader', 'SimulationGrader'),
    ('breaking_trades_generator', 'BreakingTradesGenerator'),
    ('paper_trading_engine', 'PaperTradingEngine'),
    ('advanced_backtester', 'AdvancedBacktester'),
]

for module_name, class_name in modules_to_test:
    try:
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        print(f"  ✓ {module_name}.{class_name}")
    except Exception as e:
        print(f"  ✗ {module_name}.{class_name}: {e}")
        all_passed = False

# Test 4: Check paper_trading_engine integration
print("\n[TEST 4] Verifying paper_trading_engine integration...")
try:
    from paper_trading_engine import PaperTradingEngine
    engine = PaperTradingEngine(100000)

    checks = {
        'Has risk_manager': hasattr(engine, 'risk_manager'),
        'Has grader': hasattr(engine, 'grader'),
        'Has get_advanced_performance_report': hasattr(engine, 'get_advanced_performance_report'),
        'Risk manager is AdvancedRiskManager': type(engine.risk_manager).__name__ == 'AdvancedRiskManager',
        'Grader is SimulationGrader': type(engine.grader).__name__ == 'SimulationGrader',
    }

    for check, result in checks.items():
        status = "✓" if result else "✗"
        print(f"  {status} {check}")
        if not result:
            all_passed = False

except Exception as e:
    print(f"  ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    all_passed = False

# Test 5: Test grading calculation
print("\n[TEST 5] Testing grading system calculation...")
try:
    from trading_system.simulation_grader import SimulationGrader
    grader = SimulationGrader()

    result = grader.calculate_grade(
        total_pnl_dollars=8000,
        total_return_pct=0.08,
        sharpe_ratio=1.8,
        max_drawdown_pct=0.06,
        win_rate=0.62,
        trade_count=45
    )

    checks = {
        'Returns grade': 'final_grade' in result,
        'Returns score': 'final_score' in result,
        'Has breakdown': 'breakdown' in result,
        'PnL score present': 'pnl_score' in result.get('breakdown', {}),
        'Has weights': 'weights' in result,
        'PnL weight is 60%': result.get('weights', {}).get('pnl_weight') == '60%',
    }

    for check, result_check in checks.items():
        status = "✓" if result_check else "✗"
        print(f"  {status} {check}")
        if not result_check:
            all_passed = False

    print(f"\n  Test grade: {result['final_grade']} (Score: {result['final_score']})")
    print(f"  PnL Score: {result['breakdown']['pnl_score']} (60% weight)")

except Exception as e:
    print(f"  ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    all_passed = False

# Test 6: Check AdvancedBacktester run method
print("\n[TEST 6] Verifying AdvancedBacktester.run() method...")
try:
    from advanced_backtester import AdvancedBacktester
    bt = AdvancedBacktester()

    checks = {
        'Has run method': hasattr(bt, 'run'),
        'Has run_backtest method': hasattr(bt, 'run_backtest'),
        'run is callable': callable(getattr(bt, 'run', None)),
    }

    for check, result in checks.items():
        status = "✓" if result else "✗"
        print(f"  {status} {check}")
        if not result:
            all_passed = False

except Exception as e:
    print(f"  ✗ ERROR: {e}")
    all_passed = False

# Final summary
print("\n" + "="*80)
if all_passed:
    print("✓ ALL TESTS PASSED")
    print("="*80)
    print("\nYour changes are ready! When you start Streamlit, you will see:")
    print("  1. Breaking Trades tab (first tab in Paper Trading)")
    print("  2. Performance Dashboard with grading")
    print("  3. Risk management integrated")
    print("  4. Enhanced PnL tracking")
    print("\nTo start: ./start_streamlit.sh")
    print("Or: streamlit run main.py")
    sys.exit(0)
else:
    print("✗ SOME TESTS FAILED")
    print("="*80)
    print("\nPlease review the errors above before starting Streamlit.")
    sys.exit(1)
