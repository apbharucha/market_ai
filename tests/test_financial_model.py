import unittest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from financial_model_generator import FinancialModelGenerator

class TestFinancialModel(unittest.TestCase):
    def test_dcf(self):
        gen = FinancialModelGenerator()
        # Simple inputs
        # FCF=100, Growth=10%, Term=2%, WACC=10%, Shares=1000, Debt=500
        res = gen.generate_dcf("TEST", 100.0, 0.10, 0.02, 0.10, 1000.0, 500.0)
        
        self.assertEqual(res['type'], "DCF")
        self.assertEqual(res['ticker'], "TEST")
        
        # Fair value should be positive
        self.assertTrue(res['fair_value'] > 0)
        
        # Check integrity
        # Enterprise Value = Equity Value + Debt
        self.assertAlmostEqual(res['enterprise_value'], res['equity_value'] + 500.0)
        
if __name__ == '__main__':
    unittest.main()
