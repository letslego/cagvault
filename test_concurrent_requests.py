#!/usr/bin/env python3
"""
Test script for concurrent request handling in CagVault.

This script validates that Ollama can handle multiple simultaneous requests
and that the configuration is correct.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from models import create_llm

def simple_query(model, query_id: int) -> dict:
    """Send a simple query to the LLM."""
    start = time.time()
    try:
        llm = create_llm(Config.MODEL)
        response = llm.invoke(f"What is {query_id} + {query_id}? Just give the number.")
        duration = time.time() - start
        return {
            "query_id": query_id,
            "success": True,
            "duration": duration,
            "response": response.content[:50]  # First 50 chars
        }
    except Exception as e:
        duration = time.time() - start
        return {
            "query_id": query_id,
            "success": False,
            "duration": duration,
            "error": str(e)
        }

def test_sequential(num_queries: int = 4):
    """Test sequential query execution."""
    print(f"\n🔄 Testing {num_queries} SEQUENTIAL queries...")
    llm = create_llm(Config.MODEL)
    
    start = time.time()
    results = []
    for i in range(num_queries):
        result = simple_query(None, i + 1)
        results.append(result)
        print(f"  Query {i+1}: {result['duration']:.2f}s - {'✓' if result['success'] else '✗'}")
    
    total_time = time.time() - start
    avg_time = total_time / num_queries
    
    print(f"\n  Total time: {total_time:.2f}s")
    print(f"  Average per query: {avg_time:.2f}s")
    
    return total_time, results

def test_concurrent(num_queries: int = 4):
    """Test concurrent query execution."""
    print(f"\n⚡ Testing {num_queries} CONCURRENT queries (max_workers={Config.OLLAMA_NUM_PARALLEL})...")
    
    start = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=Config.OLLAMA_NUM_PARALLEL) as executor:
        # Submit all queries at once
        futures = {
            executor.submit(simple_query, None, i + 1): i + 1 
            for i in range(num_queries)
        }
        
        # Collect results as they complete
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"  Query {result['query_id']}: {result['duration']:.2f}s - {'✓' if result['success'] else '✗'}")
    
    total_time = time.time() - start
    avg_time = total_time / num_queries
    
    print(f"\n  Total time: {total_time:.2f}s")
    print(f"  Average per query: {avg_time:.2f}s")
    
    return total_time, results

def main():
    print("=" * 60)
    print("CagVault Concurrent Request Test")
    print("=" * 60)
    
    print("\n📋 Configuration:")
    print(f"  Model: {Config.MODEL.name}")
    print(f"  Ollama URL: {Config.OLLAMA_BASE_URL}")
    print(f"  Parallel Workers: {Config.OLLAMA_NUM_PARALLEL}")
    print(f"  Request Timeout: {Config.REQUEST_TIMEOUT}s")
    print(f"  Context Window: {Config.OLLAMA_CONTEXT_WINDOW}")
    
    # Test with 4 queries
    num_queries = 4
    
    # Run sequential test
    seq_time, seq_results = test_sequential(num_queries)
    
    # Run concurrent test
    concurrent_time, concurrent_results = test_concurrent(num_queries)
    
    # Calculate speedup
    speedup = seq_time / concurrent_time if concurrent_time > 0 else 0
    
    print("\n" + "=" * 60)
    print("📊 Results Summary")
    print("=" * 60)
    print(f"  Sequential time: {seq_time:.2f}s")
    print(f"  Concurrent time: {concurrent_time:.2f}s")
    print(f"  Speedup: {speedup:.2f}x")
    print()
    
    if speedup > 1.5:
        print("✅ PASS: Concurrent requests are significantly faster!")
        print(f"   You're getting ~{speedup:.1f}x speedup with parallel processing.")
    elif speedup > 1.0:
        print("⚠️  PARTIAL: Some speedup observed, but not optimal.")
        print("   Consider adjusting OLLAMA_NUM_PARALLEL in config.py")
    else:
        print("❌ FAIL: No speedup from concurrent requests.")
        print("   Possible issues:")
        print("   - Ollama may not be configured for parallel requests")
        print("   - System may be resource-constrained (CPU/RAM)")
        print("   - Try reducing OLLAMA_NUM_PARALLEL")
    
    # Success count
    seq_success = sum(1 for r in seq_results if r['success'])
    concurrent_success = sum(1 for r in concurrent_results if r['success'])
    
    print(f"\n  Sequential success rate: {seq_success}/{num_queries}")
    print(f"  Concurrent success rate: {concurrent_success}/{num_queries}")
    
    if seq_success == num_queries and concurrent_success == num_queries:
        print("\n✅ All queries completed successfully!")
    else:
        print("\n⚠️  Some queries failed. Check Ollama connection and model availability.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error running test: {e}")
        import traceback
        traceback.print_exc()
