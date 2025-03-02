"""
Omnichannel Module for Retail Industry

This module provides functionality for creating and managing omnichannel 
retail experiences across physical stores, e-commerce, mobile apps, and 
social media platforms. It integrates various channels to provide a 
seamless shopping experience.

Key features:
- Channel integration tools
- Inventory synchronization across channels
- Unified customer profiles
- Cross-channel marketing campaigns
- Channel performance analytics
"""

import logging
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from src.utils.helpers import safe_request
from src.models.templates import get_template_by_platform
from src.feedback.feedback_store import store_feedback

# Configure logging
logger = logging.getLogger(__name__)

class OmnichannelManager:
    """
    Manages omnichannel marketing and retail operations across multiple platforms.
    Provides tools for creating consistent experiences across all customer touchpoints.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the OmnichannelManager with configuration.
        
        Args:
            config_path: Path to a JSON configuration file for omnichannel settings
        """
        self.channels = {
            "physical_store": {},
            "ecommerce": {},
            "mobile_app": {},
            "social_media": {},
            "customer_service": {}
        }
        
        self.channel_metrics = {}
        self.integration_points = {}
        
        if config_path:
            try:
                with open(config_path, 'r') as file:
                    config = json.load(file)
                    self.channels.update(config.get('channels', {}))
                    self.integration_points = config.get('integration_points', {})
                    logger.info(f"Loaded omnichannel configuration from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load omnichannel configuration: {str(e)}")
    
    def register_channel(self, channel_type: str, channel_info: Dict[str, Any]) -> bool:
        """
        Register a new channel or update an existing one.
        
        Args:
            channel_type: Type of channel (e.g., 'physical_store', 'ecommerce', 'mobile_app')
            channel_info: Dictionary containing channel details and configuration
            
        Returns:
            bool: True if successful, False otherwise
        """
        if channel_type not in self.channels:
            self.channels[channel_type] = {}
            
        channel_id = channel_info.get('id', f"{channel_type}_{datetime.now().timestamp()}")
        self.channels[channel_type][channel_id] = channel_info
        
        logger.info(f"Registered channel: {channel_type} with ID: {channel_id}")
        return True
    
    def create_omnichannel_content(self, content: str, target_channels: List[str], 
                                  content_type: str, audience: str = "general") -> Dict[str, Any]:
        """
        Create content adapted for multiple channels based on a single source content.
        
        Args:
            content: The base content to adapt
            target_channels: List of channel types to target
            content_type: Type of content (product, promotion, announcement, etc.)
            audience: Target audience identifier
            
        Returns:
            Dictionary containing channel-specific content versions
        """
        results = {}
        
        try:
            for channel in target_channels:
                if channel not in self.channels:
                    logger.warning(f"Channel {channel} not configured, skipping")
                    continue
                
                # Get channel-appropriate template
                template = get_template_by_platform(channel, content_type)
                
                # Format content for this specific channel
                channel_content = self._adapt_content_for_channel(content, channel, template, audience)
                
                results[channel] = {
                    "content": channel_content,
                    "timestamp": datetime.now().isoformat(),
                    "audience": audience,
                    "content_type": content_type
                }
                
                logger.info(f"Created content for channel: {channel}")
                
        except Exception as e:
            logger.error(f"Error creating omnichannel content: {str(e)}")
            
        return results
    
    def _adapt_content_for_channel(self, content: str, channel: str, 
                                  template: Dict[str, Any], audience: str) -> str:
        """
        Adapt content for a specific channel using channel-specific rules and templates.
        
        Args:
            content: Original content
            channel: Target channel
            template: Channel template
            audience: Target audience
            
        Returns:
            Adapted content for the specific channel
        """
        # Apply channel-specific transformations
        if channel == "social_media":
            # Shorter content with hashtags
            max_length = template.get("max_length", 280)
            adapted = content[:max_length]
            
            # Add hashtags appropriate for the content type
            hashtags = template.get("hashtags", [])
            if hashtags:
                adapted += "\n" + " ".join(["#" + tag for tag in hashtags])
                
        elif channel == "physical_store":
            # Format for in-store displays or print materials
            adapted = template.get("prefix", "") + content + template.get("suffix", "")
            
        elif channel == "ecommerce":
            # Optimize for online shopping with SEO keywords
            seo_keywords = template.get("seo_keywords", [])
            adapted = content
            
            if seo_keywords:
                adapted += "\n\nTags: " + ", ".join(seo_keywords)
                
        elif channel == "mobile_app":
            # Shorter, more direct content for mobile
            max_length = template.get("max_length", 400)
            adapted = content[:max_length]
            
        else:
            # Default adaptation if no specific rules
            adapted = content
            
        # Adjust tone based on audience
        if audience == "professional":
            # More formal tone for business audience
            adapted = adapted.replace("Hey", "Greetings").replace("Check out", "Discover")
        elif audience == "youth":
            # More casual, energetic tone
            adapted = adapted.replace("Discover", "Check out").replace("Purchase", "Get")
            
        return adapted
    
    def track_channel_performance(self, channel: str, metrics: Dict[str, Any]) -> None:
        """
        Track performance metrics for a specific channel.
        
        Args:
            channel: Channel identifier
            metrics: Dictionary of performance metrics
        """
        if channel not in self.channel_metrics:
            self.channel_metrics[channel] = []
            
        metrics["timestamp"] = datetime.now().isoformat()
        self.channel_metrics[channel].append(metrics)
        
        logger.info(f"Tracked metrics for channel: {channel}")
    
    def synchronize_inventory(self, inventory_updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronize inventory data across all channels.
        
        Args:
            inventory_updates: Dictionary containing inventory changes
            
        Returns:
            Dictionary with synchronization results
        """
        results = {
            "success": True,
            "channels_updated": [],
            "errors": {}
        }
        
        for channel_type in self.channels:
            for channel_id in self.channels[channel_type]:
                try:
                    # In a real implementation, this would make API calls to update
                    # inventory in external systems
                    
                    # Simulate synchronization
                    logger.info(f"Synchronizing inventory for {channel_type} channel {channel_id}")
                    
                    # Add to successful updates
                    results["channels_updated"].append(f"{channel_type}:{channel_id}")
                    
                except Exception as e:
                    logger.error(f"Error synchronizing inventory for {channel_type}:{channel_id}: {str(e)}")
                    results["success"] = False
                    results["errors"][f"{channel_type}:{channel_id}"] = str(e)
        
        return results
    
    def generate_omnichannel_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report on omnichannel performance.
        
        Returns:
            Dictionary containing report data
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "channels": {
                channel_type: len(channels) 
                for channel_type, channels in self.channels.items() if channels
            },
            "metrics": {},
            "recommendations": []
        }
        
        # Compile metrics across channels
        for channel, metrics_list in self.channel_metrics.items():
            if not metrics_list:
                continue
                
            # Calculate averages and totals
            report["metrics"][channel] = {
                "total_entries": len(metrics_list),
                "latest": metrics_list[-1],
                "averages": self._calculate_metric_averages(metrics_list)
            }
            
        # Generate recommendations based on metrics
        if self.channel_metrics:
            report["recommendations"] = self._generate_channel_recommendations()
            
        return report
    
    def _calculate_metric_averages(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate average values for numeric metrics.
        
        Args:
            metrics_list: List of metric dictionaries
            
        Returns:
            Dictionary of average values
        """
        if not metrics_list:
            return {}
            
        # Initialize with the keys from the first metrics entry
        totals = {}
        counts = {}
        
        for metrics in metrics_list:
            for key, value in metrics.items():
                # Skip non-numeric or timestamp fields
                if key == "timestamp" or not isinstance(value, (int, float)):
                    continue
                    
                totals[key] = totals.get(key, 0) + value
                counts[key] = counts.get(key, 0) + 1
        
        # Calculate averages
        averages = {
            key: total / counts[key] 
            for key, total in totals.items() if counts[key] > 0
        }
        
        return averages
    
    def _generate_channel_recommendations(self) -> List[str]:
        """
        Generate recommendations for improving omnichannel strategy.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Analyze metrics to find channels that need improvement
        low_performing = []
        high_performing = []
        
        for channel, metrics in self.channel_metrics.items():
            if not metrics:
                continue
                
            # This would have more sophisticated logic in a real implementation
            # For now, just check the latest engagement rate if it exists
            latest = metrics[-1]
            if "engagement_rate" in latest:
                if latest["engagement_rate"] < 0.02:  # 2% threshold
                    low_performing.append(channel)
                elif latest["engagement_rate"] > 0.1:  # 10% threshold
                    high_performing.append(channel)
        
        # Generate recommendations based on performance
        if low_performing:
            channels_str = ", ".join(low_performing)
            recommendations.append(
                f"Improve content strategy for low-performing channels: {channels_str}"
            )
            
        if high_performing:
            channels_str = ", ".join(high_performing)
            recommendations.append(
                f"Analyze success factors in high-performing channels: {channels_str} and apply to other channels"
            )
            
        # Generic recommendations
        if len(self.channels) < 3:
            recommendations.append(
                "Consider expanding to additional channels to reach more customers"
            )
            
        if not self.integration_points:
            recommendations.append(
                "Set up integration points between channels to improve cross-channel customer experience"
            )
            
        return recommendations

# Export a default instance for easier imports
default_omnichannel_manager = OmnichannelManager()

def create_cross_channel_campaign(campaign_name: str, base_content: str, 
                                channels: List[str], audience: str = "general") -> Dict[str, Any]:
    """
    Create a marketing campaign across multiple channels.
    
    Args:
        campaign_name: Name of the campaign
        base_content: Base content to adapt for each channel
        channels: List of channels to target
        audience: Target audience
        
    Returns:
        Dictionary with campaign details and content for each channel
    """
    try:
        # Use the default omnichannel manager
        channel_content = default_omnichannel_manager.create_omnichannel_content(
            base_content, channels, "campaign", audience
        )
        
        campaign = {
            "name": campaign_name,
            "created_at": datetime.now().isoformat(),
            "audience": audience,
            "channels": channels,
            "content": channel_content
        }
        
        logger.info(f"Created cross-channel campaign: {campaign_name}")
        return campaign
        
    except Exception as e:
        logger.error(f"Error creating cross-channel campaign: {str(e)}")
        return {
            "error": str(e),
            "name": campaign_name
        }

def synchronize_customer_data(customer_id: str, update_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronize customer data across all channels.
    
    Args:
        customer_id: Unique customer identifier
        update_data: Customer data to update
        
    Returns:
        Dictionary with synchronization results
    """
    results = {
        "customer_id": customer_id,
        "timestamp": datetime.now().isoformat(),
        "channels_updated": [],
        "errors": {}
    }
    
    try:
        # In a real implementation, this would use APIs to update customer profiles
        # across different systems (CRM, e-commerce, loyalty program, etc.)
        
        # For demonstration, we'll just log the operation
        logger.info(f"Synchronizing customer {customer_id} data across channels")
        
        # Simulate updating each channel
        for channel_type in default_omnichannel_manager.channels:
            if not default_omnichannel_manager.channels[channel_type]:
                continue
                
            for channel_id in default_omnichannel_manager.channels[channel_type]:
                try:
                    # Simulate API call to update customer data
                    logger.debug(f"Updating customer {customer_id} in {channel_type}:{channel_id}")
                    results["channels_updated"].append(f"{channel_type}:{channel_id}")
                    
                except Exception as e:
                    logger.error(f"Error updating customer in {channel_type}:{channel_id}: {str(e)}")
                    results["errors"][f"{channel_type}:{channel_id}"] = str(e)
    
    except Exception as e:
        logger.error(f"Error in customer data synchronization: {str(e)}")
        results["errors"]["general"] = str(e)
        
    results["success"] = len(results["errors"]) == 0
    return results
