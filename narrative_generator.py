"""
AI vs Market Sentiment Narrative Generator

This module generates comprehensive narratives comparing:
1. AI model analysis vs market sentiment and news
2. Decision-making implications and trading recommendations
3. Anticipation factor analysis across all model logic
4. Cross-asset and cross-sector narrative connections
5. Risk assessment and opportunity identification

Author: AI Market Team
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
from dataclasses import dataclass

from ml_analysis import get_analyzer
from advanced_news_processor import AdvancedNewsProcessor
from cross_sector_analyzer import CrossSectorAnalyzer
from database_manager import get_database_manager

@dataclass
class NarrativeComparison:
    """Comprehensive narrative comparing AI vs market sentiment."""
    symbol: str
    ai_analysis: Dict[str, Any]
    market_sentiment: Dict[str, Any]
    news_analysis: Dict[str, Any]
    cross_sector_impact: Dict[str, Any]
    narrative_text: str
    confidence_divergence: float
    decision_implications: List[str]
    risk_opportunities: Dict[str, List[str]]
    anticipation_factors: Dict[str, float]
    timestamp: datetime

class NarrativeGenerator:
    """Advanced narrative generator comparing AI analysis with market sentiment."""
    
    def __init__(self):
        self.ml_analyzer = get_analyzer()
        self.news_processor = AdvancedNewsProcessor()
        self.cross_sector_analyzer = CrossSectorAnalyzer()
        self.db_manager = get_database_manager()
        
        # Narrative templates for different scenarios
        self.narrative_templates = {
            'ai_bullish_market_bearish': {
                'opening': "Our AI models detect bullish signals that contrast sharply with current market sentiment.",
                'analysis_focus': "technical momentum, fundamental strength, and cross-asset correlations",
                'implication': "potential contrarian opportunity with asymmetric risk-reward"
            },
            'ai_bearish_market_bullish': {
                'opening': "While market sentiment remains optimistic, our AI analysis reveals concerning underlying signals.",
                'analysis_focus': "deteriorating technical indicators, cross-sector weakness, and anticipation factors",
                'implication': "elevated risk of sentiment reversal and potential downside"
            },
            'ai_neutral_market_extreme': {
                'opening': "Our models suggest a more balanced outlook despite extreme market sentiment.",
                'analysis_focus': "mixed signals across timeframes and asset classes",
                'implication': "potential mean reversion opportunity as sentiment normalizes"
            },
            'alignment_strong': {
                'opening': "AI analysis strongly aligns with current market sentiment, reinforcing the directional bias.",
                'analysis_focus': "confluent signals across technical, fundamental, and sentiment indicators",
                'implication': "high-confidence directional trade with trend continuation expected"
            },
            'alignment_weak': {
                'opening': "Moderate alignment between AI analysis and market sentiment suggests cautious positioning.",
                'analysis_focus': "mixed signals requiring careful risk management",
                'implication': "range-bound trading or wait-and-see approach recommended"
            }
        }
        
        # Decision-making frameworks
        self.decision_frameworks = {
            'high_conviction': {
                'criteria': {'confidence_threshold': 0.75, 'divergence_threshold': 0.3},
                'position_sizing': 'large',
                'time_horizon': 'medium_term',
                'risk_management': 'standard_stops'
            },
            'moderate_conviction': {
                'criteria': {'confidence_threshold': 0.6, 'divergence_threshold': 0.2},
                'position_sizing': 'medium',
                'time_horizon': 'short_to_medium',
                'risk_management': 'tight_stops'
            },
            'low_conviction': {
                'criteria': {'confidence_threshold': 0.5, 'divergence_threshold': 0.1},
                'position_sizing': 'small',
                'time_horizon': 'short_term',
                'risk_management': 'very_tight_stops'
            }
        }
    
    async def generate_comprehensive_narrative(self, symbol: str, 
                                             include_cross_sector: bool = True) -> NarrativeComparison:
        """Generate comprehensive narrative comparing AI analysis vs market sentiment."""
        try:
            # Get AI analysis
            ai_analysis = await self._get_ai_analysis(symbol)
            
            # Get market sentiment and news analysis
            market_sentiment = await self._get_market_sentiment_analysis(symbol)
            
            # Get news analysis
            news_analysis = await self._get_news_analysis(symbol)
            
            # Get cross-sector impact if requested
            cross_sector_impact = {}
            if include_cross_sector:
                cross_sector_impact = await self._get_cross_sector_impact(symbol)
            
            # Calculate confidence divergence
            confidence_divergence = self._calculate_confidence_divergence(
                ai_analysis, market_sentiment, news_analysis
            )
            
            # Generate narrative text
            narrative_text = self._generate_narrative_text(
                symbol, ai_analysis, market_sentiment, news_analysis, 
                cross_sector_impact, confidence_divergence
            )
            
            # Extract decision implications
            decision_implications = self._extract_decision_implications(
                ai_analysis, market_sentiment, confidence_divergence
            )
            
            # Identify risks and opportunities
            risk_opportunities = self._identify_risk_opportunities(
                ai_analysis, market_sentiment, news_analysis, cross_sector_impact
            )
            
            # Calculate anticipation factors
            anticipation_factors = self._calculate_anticipation_factors(
                ai_analysis, market_sentiment, news_analysis, cross_sector_impact
            )
            
            return NarrativeComparison(
                symbol=symbol,
                ai_analysis=ai_analysis,
                market_sentiment=market_sentiment,
                news_analysis=news_analysis,
                cross_sector_impact=cross_sector_impact,
                narrative_text=narrative_text,
                confidence_divergence=confidence_divergence,
                decision_implications=decision_implications,
                risk_opportunities=risk_opportunities,
                anticipation_factors=anticipation_factors,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return NarrativeComparison(
                symbol=symbol,
                ai_analysis={},
                market_sentiment={},
                news_analysis={},
                cross_sector_impact={},
                narrative_text=f"Error generating narrative: {str(e)}",
                confidence_divergence=0.0,
                decision_implications=[],
                risk_opportunities={'risks': [], 'opportunities': []},
                anticipation_factors={},
                timestamp=datetime.now()
            )
    
    async def _get_ai_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive AI analysis for the symbol."""
        try:
            # This would integrate with the existing ML analyzer
            analysis = self.ml_analyzer.generate_analysis(symbol)
            
            if analysis:
                return {
                    'signal': analysis['prediction']['signal'],
                    'confidence': analysis['prediction']['confidence'],
                    'bullish_probability': analysis['prediction'].get('bullish_prob', 0.5),
                    'technical_factors': analysis['prediction'].get('signal_factors', []),
                    'timeframe_analysis': {
                        'short_term': analysis.get('short_term', {}),
                        'medium_term': analysis.get('medium_term', {}),
                        'long_term': analysis.get('long_term', {})
                    },
                    'risk_metrics': analysis.get('historical', {}),
                    'position_recommendation': analysis.get('position', {})
                }
            else:
                return {'error': 'AI analysis unavailable'}
                
        except Exception as e:
            return {'error': f'AI analysis failed: {str(e)}'}
    
    async def _get_market_sentiment_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get market sentiment analysis from various sources."""
        try:
            # Simulate market sentiment analysis (would integrate with real sentiment APIs)
            # This would typically come from options flow, social sentiment, analyst ratings, etc.
            
            # Simulate sentiment scores
            np.random.seed(hash(symbol) % 1000)
            
            sentiment_data = {
                'overall_sentiment': np.random.uniform(-0.5, 0.5),
                'sentiment_sources': {
                    'social_media': np.random.uniform(-0.6, 0.6),
                    'analyst_ratings': np.random.uniform(-0.4, 0.4),
                    'options_flow': np.random.uniform(-0.7, 0.7),
                    'institutional_positioning': np.random.uniform(-0.3, 0.3)
                },
                'sentiment_trend': {
                    '1d': np.random.uniform(-0.2, 0.2),
                    '7d': np.random.uniform(-0.3, 0.3),
                    '30d': np.random.uniform(-0.4, 0.4)
                },
                'sentiment_strength': np.random.uniform(0.3, 0.9),
                'contrarian_indicators': {
                    'extreme_bullishness': np.random.choice([True, False], p=[0.2, 0.8]),
                    'extreme_bearishness': np.random.choice([True, False], p=[0.2, 0.8]),
                    'complacency_level': np.random.uniform(0, 1)
                }
            }
            
            return sentiment_data
            
        except Exception as e:
            return {'error': f'Market sentiment analysis failed: {str(e)}'}
    
    async def _get_news_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive news analysis for the symbol."""
        try:
            # Get news summary from the advanced news processor
            news_summary = await self.news_processor.get_symbol_news_analysis(symbol)
            
            return {
                'sentiment_score': news_summary.get('overall_sentiment', 0),
                'article_count': news_summary.get('article_count', 0),
                'key_themes': news_summary.get('key_themes', []),
                'recent_headlines': news_summary.get('recent_headlines', []),
                'market_moving_news': news_summary.get('market_moving_news', []),
                'anticipation_signals': news_summary.get('anticipation_signals', []),
                'cross_sector_mentions': news_summary.get('cross_sector_mentions', {}),
                'geopolitical_factors': news_summary.get('geopolitical_factors', [])
            }
            
        except Exception as e:
            return {'error': f'News analysis failed: {str(e)}'}
    
    async def _get_cross_sector_impact(self, symbol: str) -> Dict[str, Any]:
        """Get cross-sector impact analysis."""
        try:
            # Get cross-sector correlations
            cross_sector_analysis = await self.cross_sector_analyzer.analyze_symbol_cross_sector_impact(symbol)
            
            return {
                'sector_correlations': cross_sector_analysis.get('correlations', {}),
                'sector_rotation_impact': cross_sector_analysis.get('rotation_impact', {}),
                'cross_asset_implications': cross_sector_analysis.get('cross_asset_implications', {}),
                'geopolitical_sensitivity': cross_sector_analysis.get('geopolitical_sensitivity', {}),
                'economic_indicator_sensitivity': cross_sector_analysis.get('economic_sensitivity', {})
            }
            
        except Exception as e:
            return {'error': f'Cross-sector analysis failed: {str(e)}'}
    
    def _calculate_confidence_divergence(self, ai_analysis: Dict[str, Any], 
                                       market_sentiment: Dict[str, Any],
                                       news_analysis: Dict[str, Any]) -> float:
        """Calculate divergence between AI analysis and market sentiment."""
        try:
            # Get AI signal strength
            ai_confidence = ai_analysis.get('confidence', 0.5)
            ai_bullish_prob = ai_analysis.get('bullish_probability', 0.5)
            
            # Get market sentiment
            market_sentiment_score = market_sentiment.get('overall_sentiment', 0)
            market_bullish_prob = (market_sentiment_score + 1) / 2  # Convert to 0-1 scale
            
            # Get news sentiment
            news_sentiment_score = news_analysis.get('sentiment_score', 0)
            news_bullish_prob = (news_sentiment_score + 1) / 2
            
            # Calculate weighted market consensus
            market_consensus = (market_bullish_prob * 0.6) + (news_bullish_prob * 0.4)
            
            # Calculate divergence
            directional_divergence = abs(ai_bullish_prob - market_consensus)
            
            # Weight by AI confidence
            confidence_weighted_divergence = directional_divergence * ai_confidence
            
            return min(confidence_weighted_divergence, 1.0)
            
        except Exception as e:
            print(f"Confidence divergence calculation error: {e}")
            return 0.0
    
    def _generate_narrative_text(self, symbol: str, ai_analysis: Dict[str, Any],
                               market_sentiment: Dict[str, Any], news_analysis: Dict[str, Any],
                               cross_sector_impact: Dict[str, Any], confidence_divergence: float) -> str:
        """Generate comprehensive narrative text comparing AI vs market sentiment."""
        try:
            # Determine narrative scenario
            scenario = self._determine_narrative_scenario(ai_analysis, market_sentiment, confidence_divergence)
            template = self.narrative_templates.get(scenario, self.narrative_templates['alignment_weak'])
            
            # Extract key metrics
            ai_signal = ai_analysis.get('signal', 'NEUTRAL')
            ai_confidence = ai_analysis.get('confidence', 0.5)
            market_sentiment_score = market_sentiment.get('overall_sentiment', 0)
            news_sentiment = news_analysis.get('sentiment_score', 0)
            
            # Build narrative sections
            narrative_parts = []
            
            # Opening statement
            narrative_parts.append(f"**{symbol} Analysis: AI vs Market Sentiment**\n")
            narrative_parts.append(template['opening'])
            
            # AI Analysis Summary
            narrative_parts.append(f"\n**AI Model Analysis:**")
            narrative_parts.append(f"- Signal: {ai_signal} (Confidence: {ai_confidence:.1%})")
            narrative_parts.append(f"- Bullish Probability: {ai_analysis.get('bullish_probability', 0.5):.1%}")
            
            if ai_analysis.get('technical_factors'):
                narrative_parts.append(f"- Key Factors: {', '.join(ai_analysis['technical_factors'][:3])}")
            
            # Market Sentiment Summary
            narrative_parts.append(f"\n**Market Sentiment Analysis:**")
            sentiment_label = "Bullish" if market_sentiment_score > 0.1 else "Bearish" if market_sentiment_score < -0.1 else "Neutral"
            narrative_parts.append(f"- Overall Sentiment: {sentiment_label} ({market_sentiment_score:+.2f})")
            
            sentiment_sources = market_sentiment.get('sentiment_sources', {})
            if sentiment_sources:
                strongest_source = max(sentiment_sources.items(), key=lambda x: abs(x[1]))
                narrative_parts.append(f"- Strongest Signal: {strongest_source[0].replace('_', ' ').title()} ({strongest_source[1]:+.2f})")
            
            # News Analysis Summary
            narrative_parts.append(f"\n**News Analysis:**")
            narrative_parts.append(f"- News Sentiment: {news_sentiment:+.2f}")
            narrative_parts.append(f"- Articles Analyzed: {news_analysis.get('article_count', 0)}")
            
            key_themes = news_analysis.get('key_themes', [])
            if key_themes:
                narrative_parts.append(f"- Key Themes: {', '.join(key_themes[:3])}")
            
            # Cross-Sector Impact (if available)
            if cross_sector_impact and not cross_sector_impact.get('error'):
                narrative_parts.append(f"\n**Cross-Sector Implications:**")
                correlations = cross_sector_impact.get('sector_correlations', {})
                if correlations:
                    top_correlation = max(correlations.items(), key=lambda x: abs(x[1])) if correlations else None
                    if top_correlation:
                        narrative_parts.append(f"- Strongest Sector Correlation: {top_correlation[0]} ({top_correlation[1]:+.2f})")
            
            # Divergence Analysis
            narrative_parts.append(f"\n**Divergence Analysis:**")
            narrative_parts.append(f"- AI vs Market Divergence: {confidence_divergence:.1%}")
            
            if confidence_divergence > 0.3:
                narrative_parts.append("- **High Divergence**: Significant disagreement between AI models and market sentiment")
            elif confidence_divergence > 0.15:
                narrative_parts.append("- **Moderate Divergence**: Some disagreement worth monitoring")
            else:
                narrative_parts.append("- **Low Divergence**: AI and market sentiment generally aligned")
            
            # Strategic Implications
            narrative_parts.append(f"\n**Strategic Implications:**")
            narrative_parts.append(f"Focus on {template['analysis_focus']} suggests {template['implication']}.")
            
            # Risk Considerations
            narrative_parts.append(f"\n**Risk Considerations:**")
            if confidence_divergence > 0.25:
                narrative_parts.append("- High divergence increases uncertainty - consider smaller position sizes")
            if market_sentiment.get('contrarian_indicators', {}).get('extreme_bullishness'):
                narrative_parts.append("- Extreme bullish sentiment may indicate contrarian opportunity")
            if market_sentiment.get('contrarian_indicators', {}).get('extreme_bearishness'):
                narrative_parts.append("- Extreme bearish sentiment may indicate oversold conditions")
            
            return "\n".join(narrative_parts)
            
        except Exception as e:
            return f"Error generating narrative: {str(e)}"
    
    def _determine_narrative_scenario(self, ai_analysis: Dict[str, Any], 
                                    market_sentiment: Dict[str, Any], 
                                    confidence_divergence: float) -> str:
        """Determine which narrative scenario applies."""
        try:
            ai_bullish_prob = ai_analysis.get('bullish_probability', 0.5)
            market_sentiment_score = market_sentiment.get('overall_sentiment', 0)
            
            ai_bullish = ai_bullish_prob > 0.6
            ai_bearish = ai_bullish_prob < 0.4
            market_bullish = market_sentiment_score > 0.2
            market_bearish = market_sentiment_score < -0.2
            
            # High divergence scenarios
            if confidence_divergence > 0.25:
                if ai_bullish and market_bearish:
                    return 'ai_bullish_market_bearish'
                elif ai_bearish and market_bullish:
                    return 'ai_bearish_market_bullish'
                elif not ai_bullish and not ai_bearish:  # AI neutral
                    return 'ai_neutral_market_extreme'
            
            # Alignment scenarios
            if confidence_divergence < 0.15:
                if (ai_bullish and market_bullish) or (ai_bearish and market_bearish):
                    return 'alignment_strong'
            
            return 'alignment_weak'
            
        except Exception as e:
            return 'alignment_weak'
    
    def _extract_decision_implications(self, ai_analysis: Dict[str, Any],
                                     market_sentiment: Dict[str, Any],
                                     confidence_divergence: float) -> List[str]:
        """Extract actionable decision implications."""
        implications = []
        
        try:
            ai_confidence = ai_analysis.get('confidence', 0.5)
            ai_signal = ai_analysis.get('signal', 'NEUTRAL')
            
            # Determine conviction level
            conviction_level = self._determine_conviction_level(ai_confidence, confidence_divergence)
            framework = self.decision_frameworks[conviction_level]
            
            # Position sizing recommendation
            implications.append(f"Position Sizing: {framework['position_sizing'].replace('_', ' ').title()}")
            
            # Time horizon
            implications.append(f"Time Horizon: {framework['time_horizon'].replace('_', ' ').title()}")
            
            # Risk management
            implications.append(f"Risk Management: {framework['risk_management'].replace('_', ' ').title()}")
            
            # Signal-specific implications
            if ai_signal == 'BUY' and confidence_divergence > 0.3:
                implications.append("Contrarian bullish opportunity - market may be overly pessimistic")
            elif ai_signal == 'SELL' and confidence_divergence > 0.3:
                implications.append("Contrarian bearish signal - market may be overly optimistic")
            elif ai_signal == 'HOLD':
                implications.append("Neutral positioning - wait for clearer signals")
            
            # Divergence-specific implications
            if confidence_divergence > 0.4:
                implications.append("High divergence suggests potential catalyst-driven moves")
            elif confidence_divergence < 0.1:
                implications.append("Low divergence supports trend continuation strategies")
            
            return implications
            
        except Exception as e:
            return [f"Error extracting implications: {str(e)}"]
    
    def _determine_conviction_level(self, ai_confidence: float, confidence_divergence: float) -> str:
        """Determine conviction level based on AI confidence and divergence."""
        if ai_confidence >= 0.75 and confidence_divergence >= 0.3:
            return 'high_conviction'
        elif ai_confidence >= 0.6 and confidence_divergence >= 0.2:
            return 'moderate_conviction'
        else:
            return 'low_conviction'
    
    def _identify_risk_opportunities(self, ai_analysis: Dict[str, Any],
                                   market_sentiment: Dict[str, Any],
                                   news_analysis: Dict[str, Any],
                                   cross_sector_impact: Dict[str, Any]) -> Dict[str, List[str]]:
        """Identify specific risks and opportunities."""
        risks = []
        opportunities = []
        
        try:
            confidence_divergence = self._calculate_confidence_divergence(ai_analysis, market_sentiment, news_analysis)
            
            # Risk identification
            if confidence_divergence > 0.4:
                risks.append("High model-market divergence increases execution risk")
            
            if market_sentiment.get('contrarian_indicators', {}).get('extreme_bullishness'):
                risks.append("Extreme bullish sentiment may lead to sharp reversals")
            
            if market_sentiment.get('contrarian_indicators', {}).get('extreme_bearishness'):
                risks.append("Extreme bearish sentiment may indicate capitulation risk")
            
            if news_analysis.get('geopolitical_factors'):
                risks.append("Geopolitical factors add uncertainty to price action")
            
            if ai_analysis.get('confidence', 0) < 0.5:
                risks.append("Low AI model confidence suggests unclear technical picture")
            
            # Opportunity identification
            if confidence_divergence > 0.3 and ai_analysis.get('confidence', 0) > 0.7:
                opportunities.append("High-confidence contrarian setup with asymmetric risk-reward")
            
            if market_sentiment.get('sentiment_strength', 0) > 0.8:
                opportunities.append("Strong sentiment momentum may drive extended moves")
            
            anticipation_signals = news_analysis.get('anticipation_signals', [])
            if anticipation_signals:
                opportunities.append(f"Anticipation signals suggest upcoming catalysts: {', '.join(anticipation_signals[:2])}")
            
            cross_asset_implications = cross_sector_impact.get('cross_asset_implications', {})
            if cross_asset_implications:
                opportunities.append("Cross-asset correlations may provide hedging or amplification opportunities")
            
            market_moving_news = news_analysis.get('market_moving_news', [])
            if market_moving_news:
                opportunities.append("Recent market-moving news may create follow-through momentum")
            
            return {'risks': risks, 'opportunities': opportunities}
            
        except Exception as e:
            return {'risks': [f"Error identifying risks: {str(e)}"], 'opportunities': []}
    
    def _calculate_anticipation_factors(self, ai_analysis: Dict[str, Any],
                                      market_sentiment: Dict[str, Any],
                                      news_analysis: Dict[str, Any],
                                      cross_sector_impact: Dict[str, Any]) -> Dict[str, float]:
        """Calculate anticipation factors across different dimensions."""
        factors = {}
        
        try:
            # Technical anticipation (from AI analysis)
            technical_factors = ai_analysis.get('technical_factors', [])
            momentum_factors = [f for f in technical_factors if 'momentum' in f.lower() or 'trend' in f.lower()]
            factors['technical_momentum'] = len(momentum_factors) / max(len(technical_factors), 1)
            
            # Sentiment anticipation
            sentiment_trend = market_sentiment.get('sentiment_trend', {})
            recent_trend = sentiment_trend.get('7d', 0)
            factors['sentiment_momentum'] = min(abs(recent_trend), 1.0)
            
            # News anticipation
            anticipation_signals = news_analysis.get('anticipation_signals', [])
            factors['news_anticipation'] = min(len(anticipation_signals) / 5.0, 1.0)
            
            # Cross-sector anticipation
            sector_rotation_impact = cross_sector_impact.get('sector_rotation_impact', {})
            if sector_rotation_impact:
                rotation_strength = max(sector_rotation_impact.values()) if sector_rotation_impact.values() else 0
                factors['sector_rotation'] = min(abs(rotation_strength), 1.0)
            else:
                factors['sector_rotation'] = 0.0
            
            # Economic sensitivity anticipation
            economic_sensitivity = cross_sector_impact.get('economic_indicator_sensitivity', {})
            if economic_sensitivity:
                sensitivity_score = sum(abs(v) for v in economic_sensitivity.values()) / len(economic_sensitivity)
                factors['economic_sensitivity'] = min(sensitivity_score, 1.0)
            else:
                factors['economic_sensitivity'] = 0.0
            
            # Overall anticipation score
            factors['overall_anticipation'] = np.mean(list(factors.values()))
            
            return factors
            
        except Exception as e:
            return {'error': f"Anticipation calculation failed: {str(e)}"}
    
    async def generate_multi_symbol_narrative(self, symbols: List[str]) -> Dict[str, NarrativeComparison]:
        """Generate narratives for multiple symbols."""
        narratives = {}
        
        for symbol in symbols:
            try:
                narrative = await self.generate_comprehensive_narrative(symbol)
                narratives[symbol] = narrative
            except Exception as e:
                narratives[symbol] = NarrativeComparison(
                    symbol=symbol,
                    ai_analysis={},
                    market_sentiment={},
                    news_analysis={},
                    cross_sector_impact={},
                    narrative_text=f"Error generating narrative for {symbol}: {str(e)}",
                    confidence_divergence=0.0,
                    decision_implications=[],
                    risk_opportunities={'risks': [], 'opportunities': []},
                    anticipation_factors={},
                    timestamp=datetime.now()
                )
        
        return narratives
    
    async def get_narrative_summary(self, symbol: str) -> Dict[str, Any]:
        """Get a condensed narrative summary for quick reference."""
        try:
            narrative = await self.generate_comprehensive_narrative(symbol)
            
            return {
                'symbol': symbol,
                'ai_signal': narrative.ai_analysis.get('signal', 'NEUTRAL'),
                'ai_confidence': narrative.ai_analysis.get('confidence', 0.5),
                'market_sentiment': narrative.market_sentiment.get('overall_sentiment', 0),
                'confidence_divergence': narrative.confidence_divergence,
                'key_implication': narrative.decision_implications[0] if narrative.decision_implications else "No clear implication",
                'top_risk': narrative.risk_opportunities['risks'][0] if narrative.risk_opportunities['risks'] else "No significant risks identified",
                'top_opportunity': narrative.risk_opportunities['opportunities'][0] if narrative.risk_opportunities['opportunities'] else "No clear opportunities identified",
                'overall_anticipation': narrative.anticipation_factors.get('overall_anticipation', 0),
                'timestamp': narrative.timestamp
            }
            
        except Exception as e:
            return {'error': f"Summary generation failed: {str(e)}"}
    
    def save_narrative_to_database(self, narrative: NarrativeComparison) -> bool:
        """Save narrative comparison to database for historical analysis."""
        try:
            narrative_data = {
                'symbol': narrative.symbol,
                'ai_analysis': json.dumps(narrative.ai_analysis),
                'market_sentiment': json.dumps(narrative.market_sentiment),
                'news_analysis': json.dumps(narrative.news_analysis),
                'cross_sector_impact': json.dumps(narrative.cross_sector_impact),
                'narrative_text': narrative.narrative_text,
                'confidence_divergence': narrative.confidence_divergence,
                'decision_implications': json.dumps(narrative.decision_implications),
                'risk_opportunities': json.dumps(narrative.risk_opportunities),
                'anticipation_factors': json.dumps(narrative.anticipation_factors),
                'timestamp': narrative.timestamp
            }
            
            # Save to database (implementation depends on database schema)
            self.db_manager.execute_query(
                """INSERT INTO narrative_comparisons 
                   (symbol, ai_analysis, market_sentiment, news_analysis, cross_sector_impact,
                    narrative_text, confidence_divergence, decision_implications, 
                    risk_opportunities, anticipation_factors, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                tuple(narrative_data.values())
            )
            
            return True
            
        except Exception as e:
            print(f"Database save error: {e}")
            return False

# Usage Example and Testing
async def main():
    """Example usage of the NarrativeGenerator."""
    generator = NarrativeGenerator()
    
    # Generate narrative for a single symbol
    print("Generating narrative for AAPL...")
    narrative = await generator.generate_comprehensive_narrative('AAPL')
    print(f"\nNarrative for {narrative.symbol}:")
    print(narrative.narrative_text)
    print(f"\nConfidence Divergence: {narrative.confidence_divergence:.1%}")
    print(f"Decision Implications: {narrative.decision_implications}")
    
    # Generate summary
    print("\nGenerating summary...")
    summary = await generator.get_narrative_summary('AAPL')
    print(f"Summary: {summary}")
    
    # Generate for multiple symbols
    print("\nGenerating narratives for multiple symbols...")
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    multi_narratives = await generator.generate_multi_symbol_narrative(symbols)
    
    for symbol, narrative in multi_narratives.items():
        print(f"\n{symbol} - Divergence: {narrative.confidence_divergence:.1%}")
        print(f"AI Signal: {narrative.ai_analysis.get('signal', 'N/A')}")
        print(f"Market Sentiment: {narrative.market_sentiment.get('overall_sentiment', 0):+.2f}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())