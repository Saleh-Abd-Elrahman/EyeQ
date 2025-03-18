#!/usr/bin/env python3
"""
Test script to verify report generation works correctly.
"""

import os
import sys
import logging
from analytics.attention_analytics import AttentionAnalytics

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Test report generation."""
    # Output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    
    # Create an instance with empty data
    analytics = AttentionAnalytics()
    
    # Try to generate report
    logger.info("Attempting to generate report with empty data...")
    report_dir = analytics.export_analytics_report(output_dir)
    
    if report_dir:
        logger.info(f"Successfully generated report at: {report_dir}")
        
        # Check if report.html exists
        report_path = os.path.join(report_dir, "report.html")
        if os.path.exists(report_path):
            logger.info(f"HTML report file exists at: {report_path}")
            print(f"\nHTML report generated successfully at: {report_path}")
        else:
            logger.error(f"HTML report file does not exist at: {report_path}")
            print(f"\nHTML report file missing at: {report_path}")
    else:
        logger.error("Failed to generate report")
        print("\nReport generation failed. Check logs for details.")
    
    # Look for existing session data
    sessions = []
    for item in os.listdir(output_dir):
        if item.startswith("session_") and os.path.isdir(os.path.join(output_dir, item)):
            sessions.append(item)
    
    if sessions:
        # Try with existing session data
        session_path = os.path.join(output_dir, sessions[0])
        logger.info(f"Attempting to generate report with session data from: {session_path}")
        
        analytics_with_data = AttentionAnalytics(session_path)
        data_report_dir = analytics_with_data.export_analytics_report(output_dir)
        
        if data_report_dir:
            logger.info(f"Successfully generated report with session data at: {data_report_dir}")
            
            # Check if report.html exists
            data_report_path = os.path.join(data_report_dir, "report.html")
            if os.path.exists(data_report_path):
                logger.info(f"HTML report file exists at: {data_report_path}")
                print(f"\nHTML report with session data generated successfully at: {data_report_path}")
            else:
                logger.error(f"HTML report file does not exist at: {data_report_path}")
                print(f"\nHTML report file with session data missing at: {data_report_path}")
        else:
            logger.error("Failed to generate report with session data")
            print("\nReport generation with session data failed. Check logs for details.")
    else:
        logger.info("No session data found to test with")
        print("\nNo session data found to test with.")

if __name__ == "__main__":
    main() 