"""
Cross-Sector Correlation and Anticipation Factor Engine

This module provides:
1. Cross-sector correlation analysis connecting foreign policies, economic indicators, and market movements
2. Anticipation factor logic that predicts potential outcomes vs current market expectations
3. Sector rotation analysis and leadership identification
4. Geopolitical impact assessment across asset classes
5. Economic indicator cross-asset impact analysis

Author: AI Market Team
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from data_sources import get_stock, get_fx, get_futures_proxy
from database_manager import get_database_manager
from ml_analysis import get_analyzer

class CrossSectorAnalyzer:
    """Advanced cross-sector correlation and anticipation factor analysis."""
    
    def __init__(self):
        self.db_manager = get_database_manager()
        self.ml_analyzer = get_analyzer()
        
        # Sector ETF mappings for correlation analysis
        self.sector_etfs = {
            'technology': 'XLK',
            'finance': 'XLF',
            'healthcare': 'XLV',
            'energy': 'XLE',
            'consumer_discretionary': 'XLY',
            'consumer_staples': 'XLP',
            'industrials': 'XLI',
            'materials': 'XLB',
            'utilities': 'XLU',
            'real_estate': 'XLRE',
            'communication': 'XLC'
        }
        
        # Economic indicators and their typical market impacts
        self.economic_indicators = {
            'interest_rates': {
                'primary_impact': ['XLF', 'XLRE', 'XLU'],  # Finance, REITs, Utilities
                'secondary_impact': ['USD/JPY', 'EUR/USD', 'GLD'],
                'anticipation_window': 30  # days
            },
            'inflation': {
                'primary_impact': ['XLE', 'XLB', 'GLD', 'TIP'],  # Energy, Materials, Gold, TIPS
                'secondary_impact': ['XLF', 'XLV', 'XLP'],
                'anticipation_window': 45
            },
            'gdp_growth': {
                'primary_impact': ['XLI', 'XLY', 'XLF'],  # Industrials, Consumer Disc, Finance
                'secondary_impact': ['EEM', 'EWJ', 'FXI'],  # Emerging markets
                'anticipation_window': 60
            },
            'employment': {
                'primary_impact': ['XLY', 'XLF', 'XLI'],  # Consumer spending related
                'secondary_impact': ['USD/EUR', 'SPY', 'QQQ'],
                'anticipation_window': 30
            }
        }
        
        # Geopolitical event mappings
        self.geopolitical_mappings = {
            'us_china_trade': {
                'direct_impact': ['FXI', 'ASHR', 'SPY', 'QQQ'],
                'sector_impact': ['XLI', 'XLK', 'XLB'],  # Industrials, Tech, Materials
                'fx_impact': ['USD/CNY', 'AUD/USD', 'NZD/USD'],
                'commodity_impact': ['CL=F', 'GC=F', 'ZS=F']
            },
            'europe_policy': {
                'direct_impact': ['EWU', 'EWG', 'EWI', 'VGK'],
                'fx_impact': ['EUR/USD', 'GBP/USD', 'EUR/GBP'],
                'sector_impact': ['XLF', 'XLE', 'XLU'],
                'bond_impact': ['IEF', 'TLT', 'HYG']
            },
            'middle_east_tensions': {
                'direct_impact': ['XLE', 'USO', 'CL=F'],
                'safe_haven': ['GLD', 'TLT', 'VIX'],
                'fx_impact': ['USD/JPY', 'CHF/USD'],
                'sector_impact': ['XLI', 'XLY']  # Negative for cyclicals
            }
        }
        
        # Anticipation factor weights
        self.anticipation_weights = {
            'technical_momentum': 0.25,
            'fundamental_divergence': 0.20,
            'cross_asset_signals': 0.20,
            'sentiment_positioning': 0.15,
            'seasonal_patterns': 0.10,
            'geopolitical_risk': 0.10
        }
        
        # Correlation cache
        self.correlation_cache = {}
        self.cache_expiry = 3600  # 1 hour
    
    async def analyze_cross_sector_correlations(self, lookback_days: int = 252) -> Dict[str, Any]:
        """Comprehensive cross-sector correlation analysis."""
        try:
            # Fetch data for all sector ETFs
            sector_data = {}
            
            tasks = []
            for sector, etf in self.sector_etfs.items():
                tasks.append(self._fetch_sector_data(sector, etf, lookback_days))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, tuple) and result[1] is not None:
                    sector, data = result
                    sector_data[sector] = data
            
            if len(sector_data) < 3:
                return {'error': 'Insufficient sector data for correlation analysis'}
            
            # Calculate correlation matrix
            correlation_matrix = self._calculate_correlation_matrix(sector_data)
            
            # Identify sector rotation patterns
            rotation_analysis = self._analyze_sector_rotation(sector_data, correlation_matrix)
            
            # Cross-asset correlations (stocks, bonds, commodities, FX)
            cross_asset_correlations = await self._analyze_cross_asset_correlations(sector_data)
            
            # Leadership analysis
            leadership_analysis = self._analyze_sector_leadership(sector_data)
            
            # Risk-on/Risk-off analysis
            risk_sentiment = self._analyze_risk_sentiment(sector_data, cross_asset_correlations)
            
            return {
                'correlation_matrix': correlation_matrix,
                'sector_rotation': rotation_analysis,
                'cross_asset_correlations': cross_asset_correlations,
                'leadership_analysis': leadership_analysis,
                'risk_sentiment': risk_sentiment,
                'analysis_timestamp': datetime.now().isoformat(),
                'lookback_period': lookback_days
            }
            
        except Exception as e:
            return {'error': f'Cross-sector analysis failed: {str(e)}'}
    
    async def _fetch_sector_data(self, sector: str, etf: str, lookback_days: int) -> Tuple[str, Optional[pd.DataFrame]]:
        """Fetch data for a sector ETF."""
        try:
            period = '1y' if lookback_days <= 252 else '2y'
            df = get_stock(etf, period=period)
            
            if df is not None and not df.empty:
                # Calculate returns
                df['returns'] = df['Close'].pct_change()
                return (sector, df.tail(lookback_days))
            else:
                return (sector, None)
                
        except Exception as e:
            print(f"Error fetching {sector} data: {e}")
            return (sector, None)
    
    def _calculate_correlation_matrix(self, sector_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate comprehensive correlation matrix."""
        try:
            # Prepare returns data
            returns_data = {}
            for sector, df in sector_data.items():
                if 'returns' in df.columns:
                    returns_data[sector] = df['returns'].dropna()
            
            if len(returns_data) < 2:
                return {}
            
            # Align all series to common dates
            common_index = None
            for returns in returns_data.values():
                if common_index is None:
                    common_index = returns.index
                else:
                    common_index = common_index.intersection(returns.index)
            
            # Create aligned DataFrame
            aligned_returns = pd.DataFrame()
            for sector, returns in returns_data.items():
                aligned_returns[sector] = returns.reindex(common_index)
            
            # Calculate correlations
            correlation_matrix = aligned_returns.corr()
            
            # Rolling correlations (30-day)
            rolling_correlations = {}
            for col1 in aligned_returns.columns:
                rolling_correlations[col1] = {}
                for col2 in aligned_returns.columns:
                    if col1 != col2:
                        rolling_corr = aligned_returns[col1].rolling(30).corr(aligned_returns[col2])
                        rolling_correlations[col1][col2] = {
                            'current': rolling_corr.iloc[-1] if not rolling_corr.empty else 0,
                            'average': rolling_corr.mean() if not rolling_corr.empty else 0,
                            'trend': 'increasing' if rolling_corr.iloc[-1] > rolling_corr.iloc[-10] else 'decreasing'
                        }
            
            # Identify correlation clusters
            clusters = self._identify_correlation_clusters(correlation_matrix)
            
            return {
                'static_correlations': correlation_matrix.to_dict(),
                'rolling_correlations': rolling_correlations,
                'correlation_clusters': clusters,
                'average_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean(),
                'correlation_range': {
                    'min': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].min(),
                    'max': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].max()
                }
            }
            
        except Exception as e:
            print(f"Correlation matrix calculation error: {e}")
            return {}
    
    def _identify_correlation_clusters(self, corr_matrix: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify sectors that move together (correlation clusters)."""
        try:
            clusters = []
            threshold = 0.7  # High correlation threshold
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > threshold:
                        high_corr_pairs.append({
                            'sector1': corr_matrix.columns[i],
                            'sector2': corr_matrix.columns[j],
                            'correlation': corr_val,
                            'relationship': 'positive' if corr_val > 0 else 'negative'
                        })
            
            # Group into clusters
            if high_corr_pairs:
                # Simple clustering - group sectors that appear together frequently
                sector_connections = {}
                for pair in high_corr_pairs:
                    s1, s2 = pair['sector1'], pair['sector2']
                    if s1 not in sector_connections:
                        sector_connections[s1] = []
                    if s2 not in sector_connections:
                        sector_connections[s2] = []
                    sector_connections[s1].append(s2)
                    sector_connections[s2].append(s1)
                
                # Create clusters
                visited = set()
                for sector, connections in sector_connections.items():
                    if sector not in visited:
                        cluster = [sector]
                        visited.add(sector)
                        for connected in connections:
                            if connected not in visited:
                                cluster.append(connected)
                                visited.add(connected)
                        
                        if len(cluster) > 1:
                            clusters.append({
                                'sectors': cluster,
                                'cluster_type': 'high_correlation',
                                'average_correlation': np.mean([
                                    corr_matrix.loc[s1, s2] for s1 in cluster for s2 in cluster if s1 != s2
                                ])
                            })
            
            return clusters
            
        except Exception as e:
            print(f"Cluster identification error: {e}")
            return []
    
    def _analyze_sector_rotation(self, sector_data: Dict[str, pd.DataFrame], 
                                correlation_matrix: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sector rotation patterns."""
        try:
            rotation_analysis = {}
            
            # Calculate recent performance for each sector
            sector_performance = {}
            for sector, df in sector_data.items():
                if len(df) >= 20:
                    # 20-day performance
                    recent_return = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100
                    sector_performance[sector] = recent_return
            
            if not sector_performance:
                return {}
            
            # Identify leaders and laggards
            sorted_performance = sorted(sector_performance.items(), key=lambda x: x[1], reverse=True)
            
            leaders = sorted_performance[:3]  # Top 3
            laggards = sorted_performance[-3:]  # Bottom 3
            
            # Rotation momentum (comparing recent vs longer-term performance)
            rotation_momentum = {}
            for sector, df in sector_data.items():
                if len(df) >= 60:
                    recent_20d = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100
                    longer_60d = (df['Close'].iloc[-1] / df['Close'].iloc[-60] - 1) * 100
                    
                    # Momentum = recent performance - longer-term performance
                    momentum = recent_20d - (longer_60d / 3)  # Normalize 60d to 20d equivalent
                    rotation_momentum[sector] = momentum
            
            # Identify rotation signals
            rotation_signals = []
            for sector, momentum in rotation_momentum.items():
                if momentum > 5:  # Strong positive momentum
                    rotation_signals.append({
                        'sector': sector,
                        'signal': 'ROTATING_IN',
                        'momentum': momentum
                    })
                elif momentum < -5:  # Strong negative momentum
                    rotation_signals.append({
                        'sector': sector,
                        'signal': 'ROTATING_OUT',
                        'momentum': momentum
                    })
            
            return {
                'sector_performance_20d': sector_performance,
                'leaders': [{'sector': s[0], 'performance': s[1]} for s in leaders],
                'laggards': [{'sector': s[0], 'performance': s[1]} for s in laggards],
                'rotation_momentum': rotation_momentum,
                'rotation_signals': rotation_signals,
                'rotation_strength': np.std(list(sector_performance.values())) if sector_performance else 0
            }
            
        except Exception as e:
            print(f"Sector rotation analysis error: {e}")
            return {}
    
    async def _analyze_cross_asset_correlations(self, sector_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze correlations across different asset classes."""
        try:
            cross_asset_data = {}
            
            # Fetch additional asset class data
            asset_symbols = {
                'bonds': 'TLT',      # 20+ Year Treasury Bond ETF
                'gold': 'GLD',       # Gold ETF
                'dollar': 'UUP',     # US Dollar ETF
                'vix': '^VIX',       # Volatility Index
                'oil': 'USO',        # Oil ETF
                'emerging_markets': 'EEM'  # Emerging Markets ETF
            }
            
            # Fetch cross-asset data
            tasks = []
            for asset_class, symbol in asset_symbols.items():
                tasks.append(self._fetch_cross_asset_data(asset_class, symbol))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, tuple) and result[1] is not None:
                    asset_class, data = result
                    cross_asset_data[asset_class] = data
            
            # Calculate cross-asset correlations
            correlations = {}
            
            # Get average sector performance
            if sector_data:
                sector_returns = {}
                for sector, df in sector_data.items():
                    if 'returns' in df.columns:
                        sector_returns[sector] = df['returns'].dropna()
                
                # Calculate average sector return
                if sector_returns:
                    # Align all series
                    common_dates = None
                    for returns in sector_returns.values():
                        if common_dates is None:
                            common_dates = returns.index
                        else:
                            common_dates = common_dates.intersection(returns.index)
                    
                    if len(common_dates) > 50:  # Need sufficient data
                        avg_sector_returns = pd.Series(0, index=common_dates)
                        for returns in sector_returns.values():
                            avg_sector_returns += returns.reindex(common_dates).fillna(0)
                        avg_sector_returns /= len(sector_returns)
                        
                        # Calculate correlations with other assets
                        for asset_class, df in cross_asset_data.items():
                            if 'returns' in df.columns:
                                asset_returns = df['returns'].reindex(common_dates).fillna(0)
                                if len(asset_returns) > 0:
                                    correlation = avg_sector_returns.corr(asset_returns)
                                    correlations[asset_class] = correlation
            
            # Risk-on/Risk-off analysis
            risk_sentiment = self._analyze_cross_asset_risk_sentiment(cross_asset_data, correlations)
            
            return {
                'cross_asset_correlations': correlations,
                'risk_sentiment': risk_sentiment,
                'asset_performance': {
                    asset: float(df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100
                    for asset, df in cross_asset_data.items()
                    if len(df) >= 20
                }
            }
            
        except Exception as e:
            print(f"Cross-asset correlation analysis error: {e}")
            return {}
    
    async def _fetch_cross_asset_data(self, asset_class: str, symbol: str) -> Tuple[str, Optional[pd.DataFrame]]:
        """Fetch data for cross-asset analysis."""
        try:
            if symbol.startswith('^'):
                # Handle special symbols like VIX
                df = get_stock(symbol, period='1y')
            else:
                df = get_stock(symbol, period='1y')
            
            if df is not None and not df.empty:
                df['returns'] = df['Close'].pct_change()
                return (asset_class, df.tail(252))  # Last year
            else:
                return (asset_class, None)
                
        except Exception as e:
            print(f"Error fetching {asset_class} data: {e}")
            return (asset_class, None)
    
    def _analyze_cross_asset_risk_sentiment(self, cross_asset_data: Dict[str, pd.DataFrame], 
                                          correlations: Dict[str, float]) -> Dict[str, Any]:
        """Analyze risk-on/risk-off sentiment from cross-asset behavior."""
        try:
            risk_indicators = {}
            
            # VIX analysis
            if 'vix' in cross_asset_data:
                vix_df = cross_asset_data['vix']
                if len(vix_df) >= 20:
                    current_vix = vix_df['Close'].iloc[-1]
                    vix_20d_avg = vix_df['Close'].tail(20).mean()
                    
                    if current_vix > vix_20d_avg * 1.2:
                        risk_indicators['vix_signal'] = 'RISK_OFF'
                    elif current_vix < vix_20d_avg * 0.8:
                        risk_indicators['vix_signal'] = 'RISK_ON'
                    else:
                        risk_indicators['vix_signal'] = 'NEUTRAL'
            
            # Dollar strength analysis
            if 'dollar' in cross_asset_data:
                dollar_df = cross_asset_data['dollar']
                if len(dollar_df) >= 20:
                    dollar_return = (dollar_df['Close'].iloc[-1] / dollar_df['Close'].iloc[-20] - 1) * 100
                    
                    if dollar_return > 2:
                        risk_indicators['dollar_signal'] = 'RISK_OFF'  # Strong dollar often risk-off
                    elif dollar_return < -2:
                        risk_indicators['dollar_signal'] = 'RISK_ON'
                    else:
                        risk_indicators['dollar_signal'] = 'NEUTRAL'
            
            # Gold analysis
            if 'gold' in cross_asset_data:
                gold_df = cross_asset_data['gold']
                if len(gold_df) >= 20:
                    gold_return = (gold_df['Close'].iloc[-1] / gold_df['Close'].iloc[-20] - 1) * 100
                    
                    if gold_return > 3:
                        risk_indicators['gold_signal'] = 'RISK_OFF'  # Gold rally often risk-off
                    elif gold_return < -3:
                        risk_indicators['gold_signal'] = 'RISK_ON'
                    else:
                        risk_indicators['gold_signal'] = 'NEUTRAL'
            
            # Bonds analysis
            if 'bonds' in cross_asset_data:
                bonds_df = cross_asset_data['bonds']
                if len(bonds_df) >= 20:
                    bonds_return = (bonds_df['Close'].iloc[-1] / bonds_df['Close'].iloc[-20] - 1) * 100
                    
                    if bonds_return > 2:
                        risk_indicators['bonds_signal'] = 'RISK_OFF'  # Bond rally often risk-off
                    elif bonds_return < -2:
                        risk_indicators['bonds_signal'] = 'RISK_ON'
                    else:
                        risk_indicators['bonds_signal'] = 'NEUTRAL'
            
            # Overall risk sentiment
            risk_off_count = sum(1 for signal in risk_indicators.values() if signal == 'RISK_OFF')
            risk_on_count = sum(1 for signal in risk_indicators.values() if signal == 'RISK_ON')
            
            if risk_off_count > risk_on_count:
                overall_sentiment = 'RISK_OFF'
            elif risk_on_count > risk_off_count:
                overall_sentiment = 'RISK_ON'
            else:
                overall_sentiment = 'NEUTRAL'
            
            return {
                'individual_indicators': risk_indicators,
                'overall_sentiment': overall_sentiment,
                'risk_off_signals': risk_off_count,
                'risk_on_signals': risk_on_count,
                'sentiment_strength': abs(risk_off_count - risk_on_count) / max(len(risk_indicators), 1)
            }
            
        except Exception as e:
            print(f"Risk sentiment analysis error: {e}")
            return {}
    
    def _analyze_sector_leadership(self, sector_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze sector leadership patterns."""
        try:
            leadership_analysis = {}
            
            # Calculate multiple timeframe performance
            timeframes = [5, 10, 20, 60]  # days
            
            for timeframe in timeframes:
                performance = {}
                for sector, df in sector_data.items():
                    if len(df) >= timeframe:
                        perf = (df['Close'].iloc[-1] / df['Close'].iloc[-timeframe] - 1) * 100
                        performance[sector] = perf
                
                if performance:
                    # Sort by performance
                    sorted_perf = sorted(performance.items(), key=lambda x: x[1], reverse=True)
                    leadership_analysis[f'{timeframe}d_leaders'] = sorted_perf[:3]
                    leadership_analysis[f'{timeframe}d_laggards'] = sorted_perf[-3:]
            
            # Identify consistent leaders/laggards
            all_leaders = []
            all_laggards = []
            
            for timeframe in timeframes:
                leaders_key = f'{timeframe}d_leaders'
                laggards_key = f'{timeframe}d_laggards'
                
                if leaders_key in leadership_analysis:
                    all_leaders.extend([sector[0] for sector in leadership_analysis[leaders_key]])
                if laggards_key in leadership_analysis:
                    all_laggards.extend([sector[0] for sector in leadership_analysis[laggards_key]])
            
            # Count occurrences
            leader_counts = {}
            laggard_counts = {}
            
            for sector in all_leaders:
                leader_counts[sector] = leader_counts.get(sector, 0) + 1
            
            for sector in all_laggards:
                laggard_counts[sector] = laggard_counts.get(sector, 0) + 1
            
            # Consistent leaders/laggards (appear in multiple timeframes)
            consistent_leaders = [sector for sector, count in leader_counts.items() if count >= 2]
            consistent_laggards = [sector for sector, count in laggard_counts.items() if count >= 2]
            
            leadership_analysis['consistent_leaders'] = consistent_leaders
            leadership_analysis['consistent_laggards'] = consistent_laggards
            
            return leadership_analysis
            
        except Exception as e:
            print(f"Sector leadership analysis error: {e}")
            return {}
    
    def _analyze_risk_sentiment(self, sector_data: Dict[str, pd.DataFrame], 
                               cross_asset_correlations: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall market risk sentiment."""
        try:
            risk_analysis = {}
            
            # Defensive vs Cyclical performance
            defensive_sectors = ['utilities', 'consumer_staples', 'healthcare']
            cyclical_sectors = ['technology', 'finance', 'industrials', 'materials']
            
            defensive_performance = []
            cyclical_performance = []
            
            for sector, df in sector_data.items():
                if len(df) >= 20:
                    perf = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) * 100
                    
                    if sector in defensive_sectors:
                        defensive_performance.append(perf)
                    elif sector in cyclical_sectors:
                        cyclical_performance.append(perf)
            
            if defensive_performance and cyclical_performance:
                avg_defensive = np.mean(defensive_performance)
                avg_cyclical = np.mean(cyclical_performance)
                
                # Risk sentiment based on defensive vs cyclical
                if avg_defensive > avg_cyclical + 2:
                    risk_sentiment = 'RISK_OFF'
                elif avg_cyclical > avg_defensive + 2:
                    risk_sentiment = 'RISK_ON'
                else:
                    risk_sentiment = 'NEUTRAL'
                
                risk_analysis['defensive_vs_cyclical'] = {
                    'defensive_avg': avg_defensive,
                    'cyclical_avg': avg_cyclical,
                    'sentiment': risk_sentiment
                }
            
            # Cross-asset risk sentiment
            cross_asset_risk = cross_asset_correlations.get('risk_sentiment', {})
            if cross_asset_risk:
                risk_analysis['cross_asset_sentiment'] = cross_asset_risk.get('overall_sentiment', 'NEUTRAL')
            
            # Overall risk assessment
            sentiments = []
            if 'defensive_vs_cyclical' in risk_analysis:
                sentiments.append(risk_analysis['defensive_vs_cyclical']['sentiment'])
            if 'cross_asset_sentiment' in risk_analysis:
                sentiments.append(risk_analysis['cross_asset_sentiment'])
            
            # Determine overall sentiment
            if sentiments:
                risk_off_count = sentiments.count('RISK_OFF')
                risk_on_count = sentiments.count('RISK_ON')
                
                if risk_off_count > risk_on_count:
                    overall_sentiment = 'RISK_OFF'
                elif risk_on_count > risk_off_count:
                    overall_sentiment = 'RISK_ON'
                else:
                    overall_sentiment = 'NEUTRAL'
                
                risk_analysis['overall_sentiment'] = overall_sentiment
            
            return risk_analysis
            
        except Exception as e:
            print(f"Risk sentiment analysis error: {e}")
            return {}
    
    async def analyze_symbol_cross_sector_impact(self, symbol: str) -> Dict[str, Any]:
        """Analyze cross-sector impact for a specific symbol."""
        try:
            # Determine symbol's primary sector
            symbol_sector = self._determine_symbol_sector(symbol)
            
            # Get cross-sector correlations
            cross_sector_analysis = await self.analyze_cross_sector_correlations()
            
            if cross_sector_analysis.get('error'):
                return cross_sector_analysis
            
            # Extract relevant correlations for this symbol's sector
            correlation_matrix = cross_sector_analysis.get('correlation_matrix', {})
            static_correlations = correlation_matrix.get('static_correlations', {})
            
            symbol_correlations = {}
            if symbol_sector in static_correlations:
                symbol_correlations = static_correlations[symbol_sector]
            
            # Sector rotation impact
            rotation_analysis = cross_sector_analysis.get('sector_rotation', {})
            rotation_signals = rotation_analysis.get('rotation_signals', [])
            
            # Find rotation signals for this sector
            sector_rotation_signal = 'NEUTRAL'
            for signal in rotation_signals:
                if signal['sector'] == symbol_sector:
                    sector_rotation_signal = signal['signal']
                    break
            
            # Cross-asset implications
            cross_asset_correlations = cross_sector_analysis.get('cross_asset_correlations', {})
            
            # Geopolitical sensitivity
            geopolitical_sensitivity = self._assess_geopolitical_sensitivity(symbol, symbol_sector)
            
            # Economic indicator sensitivity
            economic_sensitivity = self._assess_economic_sensitivity(symbol, symbol_sector)
            
            return {
                'symbol': symbol,
                'primary_sector': symbol_sector,
                'correlations': symbol_correlations,
                'sector_rotation_signal': sector_rotation_signal,
                'rotation_impact': {
                    'signal': sector_rotation_signal,
                    'momentum': self._get_sector_momentum(symbol_sector, rotation_analysis)
                },
                'cross_asset_implications': cross_asset_correlations,
                'geopolitical_sensitivity': geopolitical_sensitivity,
                'economic_sensitivity': economic_sensitivity,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': f'Cross-sector analysis failed for {symbol}: {str(e)}'}
    
    def _determine_symbol_sector(self, symbol: str) -> str:
        """Determine the primary sector for a symbol."""
        # Simplified sector mapping
        sector_mappings = {
            'AAPL': 'technology', 'MSFT': 'technology', 'GOOGL': 'technology',
            'AMZN': 'technology', 'META': 'technology', 'NVDA': 'technology',
            'TSLA': 'technology', 'JPM': 'finance', 'BAC': 'finance',
            'WFC': 'finance', 'GS': 'finance', 'MS': 'finance',
            'JNJ': 'healthcare', 'PFE': 'healthcare', 'UNH': 'healthcare',
            'XOM': 'energy', 'CVX': 'energy', 'COP': 'energy',
            'SPY': 'broad_market', 'QQQ': 'technology', 'IWM': 'small_cap'
        }
        
        return sector_mappings.get(symbol, 'general')
    
    def _assess_geopolitical_sensitivity(self, symbol: str, sector: str) -> Dict[str, Any]:
        """Assess geopolitical sensitivity for symbol/sector."""
        try:
            # Geopolitical sensitivity by sector
            sector_sensitivity = {
                'technology': {'level': 'HIGH', 'factors': ['trade_wars', 'china_relations', 'regulation']},
                'energy': {'level': 'VERY_HIGH', 'factors': ['middle_east', 'russia_sanctions', 'climate_policy']},
                'finance': {'level': 'MEDIUM', 'factors': ['regulation', 'interest_rates', 'global_stability']},
                'healthcare': {'level': 'MEDIUM', 'factors': ['regulation', 'drug_pricing', 'trade_policy']},
                'industrials': {'level': 'HIGH', 'factors': ['trade_wars', 'infrastructure', 'defense_spending']},
                'materials': {'level': 'HIGH', 'factors': ['trade_wars', 'commodity_policy', 'environmental_regulation']}
            }
            
            sensitivity = sector_sensitivity.get(sector, {'level': 'MEDIUM', 'factors': ['general_policy']})
            
            # Symbol-specific adjustments
            if symbol in ['AAPL', 'GOOGL', 'MSFT']:  # Major tech companies
                sensitivity['level'] = 'VERY_HIGH'
                sensitivity['factors'].extend(['antitrust', 'data_privacy'])
            
            return sensitivity
            
        except Exception as e:
            return {'level': 'MEDIUM', 'factors': ['unknown'], 'error': str(e)}
    
    def _assess_economic_sensitivity(self, symbol: str, sector: str) -> Dict[str, Any]:
        """Assess economic indicator sensitivity."""
        try:
            # Economic sensitivity by sector
            sector_sensitivity = {
                'technology': {
                    'primary_indicators': ['gdp_growth', 'consumer_spending'],
                    'sensitivity_level': 'HIGH'
                },
                'finance': {
                    'primary_indicators': ['interest_rates', 'yield_curve', 'credit_spreads'],
                    'sensitivity_level': 'VERY_HIGH'
                },
                'energy': {
                    'primary_indicators': ['oil_prices', 'industrial_production'],
                    'sensitivity_level': 'HIGH'
                },
                'consumer_discretionary': {
                    'primary_indicators': ['consumer_confidence', 'employment', 'disposable_income'],
                    'sensitivity_level': 'VERY_HIGH'
                },
                'utilities': {
                    'primary_indicators': ['interest_rates', 'inflation'],
                    'sensitivity_level': 'MEDIUM'
                }
            }
            
            return sector_sensitivity.get(sector, {
                'primary_indicators': ['gdp_growth'],
                'sensitivity_level': 'MEDIUM'
            })
            
        except Exception as e:
            return {'primary_indicators': ['unknown'], 'sensitivity_level': 'MEDIUM', 'error': str(e)}
    
    def _get_sector_momentum(self, sector: str, rotation_analysis: Dict[str, Any]) -> float:
        """Get momentum score for a sector."""
        try:
            rotation_momentum = rotation_analysis.get('rotation_momentum', {})
            return rotation_momentum.get(sector, 0.0)
        except Exception:
            return 0.0
    
    def _calculate_fx_pivot_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate FX pivot levels."""
        try:
            if len(df) < 2:
                return {}
            
            yesterday = df.iloc[-2]
            high = yesterday['High']
            low = yesterday['Low']
            close = yesterday['Close']
            
            # Standard pivot calculation
            pivot = (high + low + close) / 3
            
            return {
                'pivot': float(pivot),
                'r1': float(2 * pivot - low),
                's1': float(2 * pivot - high),
                'r2': float(pivot + (high - low)),
                's2': float(pivot - (high - low))
            }
            
        except Exception as e:
            print(f"FX pivot calculation error: {e}")
            return {}

# Global instance
_cross_sector_analyzer = None

def get_cross_sector_analyzer() -> CrossSectorAnalyzer:
    """Get or create cross-sector analyzer instance."""
    global _cross_sector_analyzer
    if _cross_sector_analyzer is None:
        _cross_sector_analyzer = CrossSectorAnalyzer()
    return _cross_sector_analyzer