#!/usr/bin/env python3
"""Display results summary"""

import json
import os

results_file = "results/inference_results.json"

if os.path.exists(results_file):
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    stats = data['statistics']
    results = data['results']
    
    print("\n" + "="*70)
    print("🎉 VANET MISBEHAVIOR DETECTION - DATASET RESULTS")
    print("="*70)
    print(f"\n📊 PROCESSING SUMMARY")
    print("-"*70)
    print(f"Total Vehicles Processed:     {stats['total_detections']:,}")
    print(f"Normal Detections:            {stats['normal_detections']:,}")
    print(f"Malicious Detections:         {stats['malicious_detections']:,}")
    print(f"Detection Rate:               {stats['detection_rate']*100:.2f}%")
    print(f"Blockchain Logs:              {stats['blockchain_logs']:,}")
    print(f"Avg Blockchain Latency:       {stats.get('average_blockchain_latency', 0):.3f}s")
    
    # Confidence statistics
    confidences = [r['confidence_percent'] for r in results]
    print(f"\n📈 CONFIDENCE STATISTICS")
    print("-"*70)
    print(f"Average Confidence:           {sum(confidences)/len(confidences):.2f}%")
    print(f"Min Confidence:               {min(confidences):.2f}%")
    print(f"Max Confidence:               {max(confidences):.2f}%")
    
    # Detection time statistics
    detection_times = [r.get('detection_time_ms', 0) for r in results]
    if detection_times:
        print(f"\n⚡ PERFORMANCE")
        print("-"*70)
        print(f"Avg Detection Time:          {sum(detection_times)/len(detection_times):.3f}ms")
        print(f"Min Detection Time:          {min(detection_times):.3f}ms")
        print(f"Max Detection Time:          {max(detection_times):.3f}ms")
    
    # Malicious detections
    malicious = [r for r in results if r['is_malicious']]
    if malicious:
        print(f"\n🚨 MALICIOUS DETECTIONS ({len(malicious)})")
        print("-"*70)
        for i, r in enumerate(malicious[:10], 1):
            blockchain_status = "✅ Logged" if r.get('blockchain_logged') else "❌ Not logged"
            print(f"{i:2d}. {r['vehicle_id']:15s} | Confidence: {r['confidence_percent']:6.2f}% | "
                  f"Type: {r.get('misbehavior_type_name', 'N/A'):15s} | {blockchain_status}")
        if len(malicious) > 10:
            print(f"    ... and {len(malicious) - 10} more")
    else:
        print(f"\n✅ ALL VEHICLES CLASSIFIED AS NORMAL")
        print("-"*70)
        print("No malicious behavior detected in this sample.")
        print("This could indicate:")
        print("  - The dataset contains mostly normal traffic")
        print("  - The model is working correctly (low false positives)")
        print("  - Feature mapping may need adjustment for real VeReMi data")
    
    print(f"\n📁 FILES GENERATED")
    print("-"*70)
    print(f"Results JSON:                 results/inference_results.json")
    print(f"Visualizations:               results/visualizations/")
    
    viz_files = [
        "1_detection_distribution.png",
        "2_confidence_distribution.png", 
        "4_performance_dashboard.png",
        "5_time_series.png",
        "6_confidence_heatmap.png"
    ]
    
    for viz in viz_files:
        if os.path.exists(f"results/visualizations/{viz}"):
            size = os.path.getsize(f"results/visualizations/{viz}")
            print(f"  ✅ {viz} ({size/1024:.1f} KB)")
    
    print("\n" + "="*70)
    print("✅ Processing Complete!")
    print("="*70)
    print("\n💡 View visualizations:")
    print("   open results/visualizations/")
    print("\n")
else:
    print("❌ Results file not found. Run inference first.")







