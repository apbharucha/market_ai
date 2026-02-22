"""
Test script for Risk Management and Simulation Grading
"""

from paper_trading_engine import PaperTradingEngine
from trading_system.risk_manager import AdvancedRiskManager, RiskLimits
from trading_system.simulation_grader import SimulationGrader
import time

def test_risk_management():
    print("\n" + "="*60)
    print("TESTING RISK MANAGEMENT")
    print("="*60)

    # Initialize engine
    engine = PaperTradingEngine(initial_capital=100000.0)
    print(f"\nInitial Capital: ${engine.initial_capital:,.2f}")

    # Test 1: Check risk limits
    print("\n--- Test 1: Risk Limits Check ---")
    risk_check = engine.risk_manager.check_trade_viability(
        symbol="AAPL",
        price=150.0,
        stop_loss=145.0,  # 3.3% stop
        entry_price=150.0
    )
    print(f"Trade Approval: {risk_check['approved']}")
    print(f"Reason: {risk_check.get('reason', 'N/A')}")
    if risk_check['approved']:
        print(f"Approved Position Size: {risk_check['position_size_units']} units")
        print(f"Position Value: ${risk_check['position_size_dollars']:,.2f}")
        print(f"Risk: ${risk_check['risk_dollars']:,.2f} ({risk_check['risk_pct']:.2%})")

    # Test 2: Execute a trade
    print("\n--- Test 2: Execute Trade ---")
    order = engine.place_order("AAPL", "BUY", 50, order_type="MARKET")
    print(f"Order Status: {order.status.value}")
    if order.status.value == "FILLED":
        print(f"Filled at: ${order.filled_price:.2f}")
        print(f"Commission: ${order.commission:.2f}")

    # Check portfolio status
    snapshot = engine.get_portfolio_snapshot()
    print(f"\nPortfolio Value: ${snapshot.total_value:,.2f}")
    print(f"Cash: ${snapshot.cash:,.2f}")
    print(f"Positions: {snapshot.positions_count}")

    # Test 3: Check risk metrics
    print("\n--- Test 3: Risk Metrics ---")
    risk_metrics = engine.risk_manager.get_portfolio_risk_metrics()
    print(f"Total Exposure: ${risk_metrics['total_exposure']:,.2f} ({risk_metrics['total_exposure_pct']:.1%})")
    print(f"Available Capital: ${risk_metrics['available_capital']:,.2f}")
    print(f"Open Positions: {risk_metrics['open_positions']}")

    # Test 4: Test risk block - too wide stop
    print("\n--- Test 4: Risk Block - Wide Stop Loss ---")
    risk_check_fail = engine.risk_manager.check_trade_viability(
        symbol="TSLA",
        price=200.0,
        stop_loss=165.0,  # 17.5% stop - should be rejected
        entry_price=200.0
    )
    print(f"Trade Approval: {risk_check_fail['approved']}")
    print(f"Reason: {risk_check_fail.get('reason', 'N/A')}")

    # Test 5: Sell and realize PnL
    print("\n--- Test 5: Close Position and Realize P&L ---")
    time.sleep(1)  # Small delay
    sell_order = engine.place_order("AAPL", "SELL", 50, order_type="MARKET")
    print(f"Sell Order Status: {sell_order.status.value}")

    snapshot2 = engine.get_portfolio_snapshot()
    print(f"Realized P&L: ${engine.realized_pnl:,.2f}")
    print(f"Portfolio Value: ${snapshot2.total_value:,.2f}")

    return engine


def test_grading_system(engine):
    print("\n" + "="*60)
    print("TESTING GRADING SYSTEM")
    print("="*60)

    # Generate performance report
    print("\n--- Generating Performance Report ---")
    report = engine.get_advanced_performance_report()

    # Display grade
    print(f"\nFINAL GRADE: {report['final_grade']}")
    print(f"Final Score: {report['final_score']}/100")
    print(f"Total P&L: {report['metrics']['pnl_dollars']}")

    # Display breakdown
    print("\n--- Score Breakdown ---")
    breakdown = report['breakdown']
    weights = report['weights']
    print(f"P&L Score: {breakdown['pnl_score']}/100 (Weight: {weights['pnl_weight']})")
    print(f"Sharpe Score: {breakdown['sharpe_score']}/100 (Weight: {weights['sharpe_weight']})")
    print(f"Drawdown Score: {breakdown['drawdown_score']}/100 (Weight: {weights['drawdown_weight']})")
    print(f"Win Rate Score: {breakdown['win_rate_score']}/100 (Weight: {weights['win_rate_weight']})")
    print(f"Consistency Score: {breakdown['consistency_score']}/100 (Weight: {weights['consistency_weight']})")

    # Display metrics
    print("\n--- Performance Metrics ---")
    metrics = report['metrics']
    print(f"Return %: {metrics['return_pct']}")
    print(f"Sharpe Ratio: {metrics['sharpe']}")
    print(f"Max Drawdown: {metrics['max_drawdown']}")
    print(f"Win Rate: {metrics['win_rate']}")
    print(f"Total Trades: {metrics['total_trades']}")

    # Display trade breakdown
    print("\n--- Recent Trades ---")
    if 'trade_breakdown' in report and report['trade_breakdown']:
        for trade in report['trade_breakdown'][-5:]:  # Last 5 trades
            print(f"Trade #{trade['trade_num']}: {trade['symbol']} {trade['side']} "
                  f"{trade['quantity']}@${trade['price']:.2f} | "
                  f"P&L: ${trade['pnl']:+.2f} ({trade['pnl_pct']:+.2f}%)")
    else:
        print("No trade breakdown available")

    # Display risk metrics
    print("\n--- Risk Management Status ---")
    risk_metrics = report['risk_metrics']
    print(f"Total Capital: ${risk_metrics['total_capital']:,.0f}")
    print(f"Total Exposure: ${risk_metrics['total_exposure']:,.0f} ({risk_metrics['total_exposure_pct']:.1%})")
    print(f"Daily P&L: ${risk_metrics['daily_pnl']:,.0f} ({risk_metrics['daily_pnl_pct']:.2%})")


def test_multiple_trades():
    print("\n" + "="*60)
    print("TESTING MULTIPLE TRADES WITH GRADING")
    print("="*60)

    engine = PaperTradingEngine(initial_capital=100000.0)

    symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]

    print("\nExecuting multiple trades...")

    for i, symbol in enumerate(symbols):
        print(f"\n--- Trade {i+1}: {symbol} ---")

        # Buy
        buy_order = engine.place_order(symbol, "BUY", 10, order_type="MARKET")
        print(f"Buy: {buy_order.status.value} @ ${buy_order.filled_price:.2f}" if buy_order.filled_price else f"Buy: {buy_order.status.value}")

        time.sleep(0.5)

        # Sell
        sell_order = engine.place_order(symbol, "SELL", 10, order_type="MARKET")
        print(f"Sell: {sell_order.status.value} @ ${sell_order.filled_price:.2f}" if sell_order.filled_price else f"Sell: {sell_order.status.value}")

    print("\n" + "="*60)
    print("FINAL GRADING REPORT")
    print("="*60)

    test_grading_system(engine)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("OCTAVIAN RISK MANAGEMENT & GRADING TEST SUITE")
    print("="*60)

    # Run tests
    try:
        engine = test_risk_management()
        test_grading_system(engine)

        print("\n\n")
        test_multiple_trades()

        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*60)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
