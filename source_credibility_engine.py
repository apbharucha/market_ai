"""
Octavian Source Credibility Engine
Advanced news source and author trustworthiness weighting system

This module provides sophisticated credibility scoring for news sources and authors
to enhance decision-making across all models and chatbot applications.

Author: APB - Octavian Team
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
from enum import Enum

class SourceTier(Enum):
    TIER_1_PREMIUM = "tier_1_premium"  # Bloomberg, Reuters, WSJ
    TIER_2_MAJOR = "tier_2_major"      # CNBC, MarketWatch, Yahoo Finance
    TIER_3_SPECIALIZED = "tier_3_specialized"  # Seeking Alpha, Benzinga
    TIER_4_SOCIAL = "tier_4_social"    # Twitter, Reddit, Forums
    TIER_5_UNKNOWN = "tier_5_unknown"  # Unknown or unverified sources

@dataclass
class SourceCredibility:
    """Comprehensive source credibility scoring."""
    source_name: str
    tier: SourceTier
    base_weight: float
    accuracy_score: float
    timeliness_score: float
    market_impact_score: float
    author_credibility: float
    final_weight: float
    last_updated: datetime

@dataclass
class AuthorCredibility:
    """Author-specific credibility metrics."""
    author_name: str
    track_record_score: float
    expertise_areas: List[str]
    prediction_accuracy: float
    market_moving_articles: int
    credibility_weight: float
    last_updated: datetime

class SourceCredibilityEngine:
    """Advanced source and author credibility weighting system."""
    
    def __init__(self):
        self.source_weights = self._initialize_source_weights()
        self.author_weights = {}
        self.credibility_history = {}
        
        # Timeframe impact multipliers
        self.timeframe_multipliers = {
            'breaking': 1.5,      # Breaking news gets higher weight
            'intraday': 1.2,      # Same day news
            'recent': 1.0,        # 1-3 days
            'weekly': 0.8,        # 1 week old
            'monthly': 0.6,       # 1 month old
            'stale': 0.3          # Older than 1 month
        }
        
        # Market session impact
        self.session_multipliers = {
            'pre_market': 1.3,
            'market_hours': 1.5,
            'after_hours': 1.2,
            'overnight': 0.9
        }
    
    def _initialize_source_weights(self) -> Dict[str, SourceCredibility]:
        """Initialize base source credibility weights."""
        sources = {
            # Tier 1 Premium Sources
            'Bloomberg': SourceCredibility(
                source_name='Bloomberg',
                tier=SourceTier.TIER_1_PREMIUM,
                base_weight=0.95,
                accuracy_score=0.92,
                timeliness_score=0.95,
                market_impact_score=0.90,
                author_credibility=0.85,
                final_weight=0.95,
                last_updated=datetime.now()
            ),
            'Reuters': SourceCredibility(
                source_name='Reuters',
                tier=SourceTier.TIER_1_PREMIUM,
                base_weight=0.93,
                accuracy_score=0.90,
                timeliness_score=0.92,
                market_impact_score=0.88,
                author_credibility=0.83,
                final_weight=0.93,
                last_updated=datetime.now()
            ),
            'Wall Street Journal': SourceCredibility(
                source_name='Wall Street Journal',
                tier=SourceTier.TIER_1_PREMIUM,
                base_weight=0.91,
                accuracy_score=0.89,
                timeliness_score=0.85,
                market_impact_score=0.92,
                author_credibility=0.88,
                final_weight=0.91,
                last_updated=datetime.now()
            ),
            
            # Tier 2 Major Sources
            'CNBC': SourceCredibility(
                source_name='CNBC',
                tier=SourceTier.TIER_2_MAJOR,
                base_weight=0.78,
                accuracy_score=0.75,
                timeliness_score=0.88,
                market_impact_score=0.82,
                author_credibility=0.70,
                final_weight=0.78,
                last_updated=datetime.now()
            ),
            'MarketWatch': SourceCredibility(
                source_name='MarketWatch',
                tier=SourceTier.TIER_2_MAJOR,
                base_weight=0.72,
                accuracy_score=0.70,
                timeliness_score=0.80,
                market_impact_score=0.75,
                author_credibility=0.68,
                final_weight=0.72,
                last_updated=datetime.now()
            ),
            'Yahoo Finance': SourceCredibility(
                source_name='Yahoo Finance',
                tier=SourceTier.TIER_2_MAJOR,
                base_weight=0.68,
                accuracy_score=0.65,
                timeliness_score=0.85,
                market_impact_score=0.70,
                author_credibility=0.60,
                final_weight=0.68,
                last_updated=datetime.now()
            ),
            
            # Tier 3 Specialized Sources
            'Seeking Alpha': SourceCredibility(
                source_name='Seeking Alpha',
                tier=SourceTier.TIER_3_SPECIALIZED,
                base_weight=0.58,
                accuracy_score=0.60,
                timeliness_score=0.70,
                market_impact_score=0.65,
                author_credibility=0.55,
                final_weight=0.58,
                last_updated=datetime.now()
            ),
            'Benzinga': SourceCredibility(
                source_name='Benzinga',
                tier=SourceTier.TIER_3_SPECIALIZED,
                base_weight=0.62,
                accuracy_score=0.58,
                timeliness_score=0.82,
                market_impact_score=0.68,
                author_credibility=0.52,
                final_weight=0.62,
                last_updated=datetime.now()
            ),
            
            # Tier 4 Social Sources
            'Twitter': SourceCredibility(
                source_name='Twitter',
                tier=SourceTier.TIER_4_SOCIAL,
                base_weight=0.35,
                accuracy_score=0.40,
                timeliness_score=0.95,
                market_impact_score=0.45,
                author_credibility=0.30,
                final_weight=0.35,
                last_updated=datetime.now()
            ),
            'Reddit': SourceCredibility(
                source_name='Reddit',
                tier=SourceTier.TIER_4_SOCIAL,
                base_weight=0.25,
                accuracy_score=0.30,
                timeliness_score=0.80,
                market_impact_score=0.35,
                author_credibility=0.20,
                final_weight=0.25,
                last_updated=datetime.now()
            )
        }
        
        return sources
    
    def calculate_weighted_news_score(self, news_item: Dict[str, Any], 
                                    timeframe_context: str = 'recent') -> float:
        """Calculate weighted news score based on source credibility and timeframe."""
        try:
            source = news_item.get('source', 'Unknown')
            author = news_item.get('author', 'Unknown')
            publish_time = news_item.get('publish_time', datetime.now())
            sentiment_score = news_item.get('sentiment_score', 0.0)
            
            # Get source credibility
            source_credibility = self.get_source_credibility(source)
            
            # Get author credibility
            author_credibility = self.get_author_credibility(author)
            
            # Calculate time decay
            time_multiplier = self.calculate_time_multiplier(publish_time, timeframe_context)
            
            # Calculate session multiplier
            session_multiplier = self.calculate_session_multiplier(publish_time)
            
            # Calculate final weighted score
            base_score = abs(sentiment_score)  # Use absolute value for impact magnitude
            
            weighted_score = (
                base_score * 
                source_credibility.final_weight * 
                author_credibility * 
                time_multiplier * 
                session_multiplier
            )
            
            # Preserve sentiment direction
            final_score = weighted_score if sentiment_score >= 0 else -weighted_score
            
            return min(max(final_score, -1.0), 1.0)  # Clamp to [-1, 1]
            
        except Exception as e:
            print(f"Error calculating weighted news score: {e}")
            return news_item.get('sentiment_score', 0.0)
    
    def get_source_credibility(self, source_name: str) -> SourceCredibility:
        """Get credibility score for a news source."""
        # Try exact match first
        if source_name in self.source_weights:
            return self.source_weights[source_name]
        
        # Try partial matching for variations
        for known_source, credibility in self.source_weights.items():
            if known_source.lower() in source_name.lower() or source_name.lower() in known_source.lower():
                return credibility
        
        # Return default for unknown sources
        return SourceCredibility(
            source_name=source_name,
            tier=SourceTier.TIER_5_UNKNOWN,
            base_weight=0.40,
            accuracy_score=0.50,
            timeliness_score=0.60,
            market_impact_score=0.40,
            author_credibility=0.50,
            final_weight=0.40,
            last_updated=datetime.now()
        )
    
    def get_author_credibility(self, author_name: str) -> float:
        """Get credibility weight for an author."""
        if author_name in self.author_weights:
            return self.author_weights[author_name].credibility_weight
        
        # Default credibility for unknown authors
        return 0.60
    
    def calculate_time_multiplier(self, publish_time: datetime, timeframe_context: str) -> float:
        """Calculate time-based multiplier for news relevance."""
        try:
            if isinstance(publish_time, str):
                publish_time = datetime.fromisoformat(publish_time.replace('Z', '+00:00'))
            
            time_diff = datetime.now() - publish_time
            hours_old = time_diff.total_seconds() / 3600
            
            # Determine time category
            if hours_old < 1:
                time_category = 'breaking'
            elif hours_old < 24:
                time_category = 'intraday'
            elif hours_old < 72:
                time_category = 'recent'
            elif hours_old < 168:  # 1 week
                time_category = 'weekly'
            elif hours_old < 720:  # 1 month
                time_category = 'monthly'
            else:
                time_category = 'stale'
            
            base_multiplier = self.timeframe_multipliers.get(time_category, 1.0)
            
            # Adjust based on timeframe context
            if timeframe_context == 'intraday' and time_category in ['breaking', 'intraday']:
                base_multiplier *= 1.2
            elif timeframe_context == 'swing' and time_category in ['recent', 'weekly']:
                base_multiplier *= 1.1
            elif timeframe_context == 'position' and time_category in ['weekly', 'monthly']:
                base_multiplier *= 1.05
            
            return base_multiplier
            
        except Exception as e:
            print(f"Error calculating time multiplier: {e}")
            return 1.0
    
    def calculate_session_multiplier(self, publish_time: datetime) -> float:
        """Calculate market session-based multiplier."""
        try:
            if isinstance(publish_time, str):
                publish_time = datetime.fromisoformat(publish_time.replace('Z', '+00:00'))
            
            hour = publish_time.hour
            
            # US market hours (EST)
            if 9 <= hour <= 16:
                return self.session_multipliers['market_hours']
            elif 4 <= hour < 9:
                return self.session_multipliers['pre_market']
            elif 16 < hour <= 20:
                return self.session_multipliers['after_hours']
            else:
                return self.session_multipliers['overnight']
                
        except Exception as e:
            print(f"Error calculating session multiplier: {e}")
            return 1.0
    
    def update_author_credibility(self, author_name: str, prediction_accuracy: float,
                                market_impact: bool = False):
        """Update author credibility based on track record."""
        if author_name not in self.author_weights:
            self.author_weights[author_name] = AuthorCredibility(
                author_name=author_name,
                track_record_score=0.60,
                expertise_areas=[],
                prediction_accuracy=0.50,
                market_moving_articles=0,
                credibility_weight=0.60,
                last_updated=datetime.now()
            )
        
        author = self.author_weights[author_name]
        
        # Update prediction accuracy with exponential moving average
        alpha = 0.1
        author.prediction_accuracy = (
            alpha * prediction_accuracy + 
            (1 - alpha) * author.prediction_accuracy
        )
        
        # Update market moving articles count
        if market_impact:
            author.market_moving_articles += 1
        
        # Recalculate credibility weight
        author.credibility_weight = min(
            0.95,
            0.4 + 0.4 * author.prediction_accuracy + 
            0.15 * min(author.market_moving_articles / 10, 1.0)
        )
        
        author.last_updated = datetime.now()
    
    def get_weighted_news_summary(self, news_items: List[Dict[str, Any]], 
                                timeframe_context: str = 'recent') -> Dict[str, Any]:
        """Get weighted summary of multiple news items."""
        if not news_items:
            return {'weighted_sentiment': 0.0, 'total_weight': 0.0, 'item_count': 0}
        
        weighted_scores = []
        total_weight = 0.0
        
        for item in news_items:
            weighted_score = self.calculate_weighted_news_score(item, timeframe_context)
            source_weight = self.get_source_credibility(item.get('source', 'Unknown')).final_weight
            
            weighted_scores.append(weighted_score)
            total_weight += source_weight
        
        # Calculate weighted average sentiment
        if total_weight > 0:
            weighted_sentiment = sum(weighted_scores) / len(weighted_scores)
        else:
            weighted_sentiment = 0.0
        
        return {
            'weighted_sentiment': weighted_sentiment,
            'total_weight': total_weight,
            'item_count': len(news_items),
            'source_breakdown': self._get_source_breakdown(news_items),
            'timeframe_context': timeframe_context
        }
    
    def _get_source_breakdown(self, news_items: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get breakdown of news items by source tier."""
        breakdown = {tier.value: 0 for tier in SourceTier}
        
        for item in news_items:
            source = item.get('source', 'Unknown')
            credibility = self.get_source_credibility(source)
            breakdown[credibility.tier.value] += 1
        
        return breakdown