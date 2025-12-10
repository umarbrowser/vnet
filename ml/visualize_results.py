"""
Generate cool visualizations from dataset inference results
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_results(results_path: str = "results/inference_results.json"):
    """Load inference results"""
    if not os.path.exists(results_path):
        logger.error(f"Results file not found: {results_path}")
        return None
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    return data

def create_visualizations(results_data: dict, output_dir: str = "results/visualizations"):
    """Create comprehensive visualizations"""
    os.makedirs(output_dir, exist_ok=True)
    
    results = results_data.get('results', [])
    stats = results_data.get('statistics', {})
    
    if not results:
        logger.error("No results to visualize")
        return
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(results)
    
    logger.info(f"Creating visualizations for {len(df)} results...")
    
    # 1. Detection Distribution Pie Chart
    fig, ax = plt.subplots(figsize=(10, 8))
    malicious_count = df['is_malicious'].sum()
    normal_count = len(df) - malicious_count
    
    colors = ['#2ecc71', '#e74c3c']
    wedges, texts, autotexts = ax.pie(
        [normal_count, malicious_count],
        labels=['Normal', 'Malicious'],
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        explode=(0.05, 0.1),
        shadow=True
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(14)
    
    ax.set_title('VANET Misbehavior Detection Distribution', fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/1_detection_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✅ Created: Detection Distribution Pie Chart")
    
    # 2. Confidence Score Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Histogram
    ax1.hist(df['confidence_percent'], bins=50, color='#3498db', edgecolor='black', alpha=0.7)
    ax1.axvline(df['confidence_percent'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["confidence_percent"].mean():.2f}%')
    ax1.set_xlabel('Confidence Score (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Confidence Score Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot by detection type
    malicious_conf = df[df['is_malicious']]['confidence_percent']
    normal_conf = df[~df['is_malicious']]['confidence_percent']
    
    bp = ax2.boxplot([normal_conf, malicious_conf], labels=['Normal', 'Malicious'], 
                     patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][1].set_facecolor('#e74c3c')
    
    ax2.set_ylabel('Confidence Score (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Confidence by Detection Type', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/2_confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✅ Created: Confidence Score Distribution")
    
    # 3. Misbehavior Types (if available)
    if 'misbehavior_type_name' in df.columns:
        misbehavior_types = df[df['is_malicious']]['misbehavior_type_name'].value_counts()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        colors_map = {'sybil': '#e74c3c', 'falsification': '#f39c12', 'replay': '#9b59b6', 'dos': '#1abc9c'}
        colors_list = [colors_map.get(mt, '#95a5a6') for mt in misbehavior_types.index]
        
        bars = ax.barh(misbehavior_types.index, misbehavior_types.values, color=colors_list, edgecolor='black')
        ax.set_xlabel('Count', fontsize=12, fontweight='bold')
        ax.set_ylabel('Misbehavior Type', fontsize=12, fontweight='bold')
        ax.set_title('Misbehavior Types Distribution', fontsize=16, fontweight='bold', pad=20)
        
        # Add value labels on bars
        for i, (idx, val) in enumerate(misbehavior_types.items()):
            ax.text(val + max(misbehavior_types.values) * 0.01, i, f'{val}', 
                   va='center', fontsize=11, fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/3_misbehavior_types.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("✅ Created: Misbehavior Types Distribution")
    
    # 4. Performance Metrics Dashboard
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Total detections
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(['Total', 'Normal', 'Malicious'], 
            [stats.get('total_detections', len(df)), 
             stats.get('normal_detections', normal_count),
             stats.get('malicious_detections', malicious_count)],
            color=['#3498db', '#2ecc71', '#e74c3c'], edgecolor='black')
    ax1.set_ylabel('Count', fontweight='bold')
    ax1.set_title('Detection Counts', fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Detection rate
    ax2 = fig.add_subplot(gs[0, 1])
    detection_rate = stats.get('detection_rate', malicious_count / len(df)) * 100
    ax2.barh(['Detection Rate'], [detection_rate], color='#9b59b6', edgecolor='black')
    ax2.set_xlabel('Percentage (%)', fontweight='bold')
    ax2.set_xlim(0, 100)
    ax2.set_title('Malicious Detection Rate', fontweight='bold')
    ax2.text(detection_rate/2, 0, f'{detection_rate:.2f}%', 
            ha='center', va='center', fontsize=14, fontweight='bold', color='white')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Blockchain logs
    ax3 = fig.add_subplot(gs[0, 2])
    blockchain_logs = stats.get('blockchain_logs', 0)
    not_logged = max(0, malicious_count - blockchain_logs)
    
    if blockchain_logs > 0 or not_logged > 0:
        ax3.pie([blockchain_logs, not_logged], 
               labels=['Logged', 'Not Logged'],
               autopct='%1.1f%%',
               colors=['#1abc9c', '#95a5a6'],
               startangle=90)
    else:
        ax3.text(0.5, 0.5, 'No malicious\ndetections', 
                ha='center', va='center', fontsize=12, fontweight='bold')
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
    ax3.set_title('Blockchain Logging', fontweight='bold')
    
    # Confidence distribution by type
    ax4 = fig.add_subplot(gs[1, :])
    if len(malicious_conf) > 0 and len(normal_conf) > 0:
        ax4.hist([normal_conf, malicious_conf], bins=30, label=['Normal', 'Malicious'],
                color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Confidence Score (%)', fontweight='bold', fontsize=12)
        ax4.set_ylabel('Frequency', fontweight='bold', fontsize=12)
        ax4.set_title('Confidence Distribution by Detection Type', fontweight='bold', fontsize=14)
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
    
    # Statistics table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    stats_text = f"""
    PERFORMANCE METRICS
    {'='*60}
    Total Detections:        {stats.get('total_detections', len(df)):,}
    Normal Detections:       {stats.get('normal_detections', normal_count):,}
    Malicious Detections:    {stats.get('malicious_detections', malicious_count):,}
    Detection Rate:          {detection_rate:.2f}%
    Blockchain Logs:         {blockchain_logs:,}
    Avg Confidence (Normal): {normal_conf.mean():.2f}% (if available)
    Avg Confidence (Malicious): {malicious_conf.mean():.2f}% (if available)
    """
    
    ax5.text(0.1, 0.5, stats_text, fontsize=12, fontfamily='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle('VANET Misbehavior Detection - Performance Dashboard', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.savefig(f'{output_dir}/4_performance_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✅ Created: Performance Dashboard")
    
    # 5. Time Series (if timestamp available)
    if 'timestamp' in df.columns or any('time' in col.lower() for col in df.columns):
        fig, ax = plt.subplots(figsize=(16, 6))
        time_col = [col for col in df.columns if 'time' in col.lower()][0] if any('time' in col.lower() for col in df.columns) else None
        
        if time_col:
            df_sorted = df.sort_values(time_col)
            ax.plot(range(len(df_sorted)), df_sorted['confidence_percent'], 
                   alpha=0.6, linewidth=1, color='#3498db', label='Confidence')
            ax.scatter(range(len(df_sorted)), df_sorted['confidence_percent'],
                      c=df_sorted['is_malicious'].map({True: '#e74c3c', False: '#2ecc71'}),
                      s=20, alpha=0.6, edgecolors='black', linewidth=0.5)
            
            ax.set_xlabel('Vehicle Index', fontweight='bold', fontsize=12)
            ax.set_ylabel('Confidence Score (%)', fontweight='bold', fontsize=12)
            ax.set_title('Confidence Score Over Time', fontweight='bold', fontsize=14)
            ax.legend(['Confidence', 'Malicious', 'Normal'], loc='upper right')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/5_time_series.png', dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("✅ Created: Time Series Analysis")
    
    # 6. Heatmap of Confidence vs Detection Type
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bins for confidence scores
    df['confidence_bin'] = pd.cut(df['confidence_percent'], bins=10, labels=[f'{i*10}-{(i+1)*10}%' for i in range(10)])
    
    if 'misbehavior_type_name' in df.columns:
        heatmap_data = pd.crosstab(df[df['is_malicious']]['confidence_bin'], 
                                  df[df['is_malicious']]['misbehavior_type_name'])
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Count'})
        ax.set_xlabel('Misbehavior Type', fontweight='bold', fontsize=12)
        ax.set_ylabel('Confidence Range', fontweight='bold', fontsize=12)
        ax.set_title('Confidence vs Misbehavior Type Heatmap', fontweight='bold', fontsize=14)
    else:
        heatmap_data = pd.crosstab(df['confidence_bin'], df['is_malicious'])
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Count'})
        ax.set_xlabel('Is Malicious', fontweight='bold', fontsize=12)
        ax.set_ylabel('Confidence Range', fontweight='bold', fontsize=12)
        ax.set_title('Confidence vs Detection Type Heatmap', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/6_confidence_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✅ Created: Confidence Heatmap")
    
    logger.info(f"\n✅ All visualizations saved to: {output_dir}/")
    logger.info("Generated visualizations:")
    logger.info("  1. Detection Distribution Pie Chart")
    logger.info("  2. Confidence Score Distribution")
    logger.info("  3. Misbehavior Types Distribution")
    logger.info("  4. Performance Dashboard")
    logger.info("  5. Time Series Analysis (if available)")
    logger.info("  6. Confidence Heatmap")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize dataset inference results')
    parser.add_argument('--results', type=str, default='results/inference_results.json',
                       help='Path to results JSON file')
    parser.add_argument('--output', type=str, default='results/visualizations',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("VANET Detection Results Visualization")
    logger.info("="*60)
    
    results_data = load_results(args.results)
    
    if results_data:
        create_visualizations(results_data, args.output)
        logger.info("\n🎉 Visualization complete!")
    else:
        logger.error("Failed to load results. Run inference first.")

