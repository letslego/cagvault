"""
Example demonstrating KV-Cache usage for Cache-Augmented Generation.

This script shows how to:
1. Create context from documents
2. Cache the context using KV-Cache
3. Retrieve cached contexts
4. Monitor cache statistics
"""

from pathlib import Path
from kvcache import get_kv_cache, KVCache


def example_basic_usage():
    """Basic KV-Cache usage example."""
    print("=" * 60)
    print("KV-Cache Basic Usage Example")
    print("=" * 60)
    
    # Initialize cache (memory-only for this example)
    cache = KVCache(memory_only=True)
    
    # Sample contexts (simulating concatenated documents)
    context_1 = """
    Document 1: Python Programming
    Python is a high-level programming language...
    """ * 100  # Make it larger for realistic token counts
    
    context_2 = """
    Document 2: Machine Learning Basics
    Machine learning is a subset of artificial intelligence...
    """ * 100
    
    # Create caches
    print("\n📝 Creating caches...")
    cache_id_1 = cache.create(context_1, source_ids=["doc1.pdf"])
    print(f"✓ Created cache: {cache_id_1}")
    
    cache_id_2 = cache.create(context_2, source_ids=["doc2.pdf"])
    print(f"✓ Created cache: {cache_id_2}")
    
    # Retrieve and display stats
    print("\n📊 Cache Statistics:")
    stats = cache.get_stats()
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Total tokens: {stats['total_tokens']}")
    
    # Display individual entries
    print("\n📦 Cache Entries:")
    for entry in stats['entries']:
        print(f"  - {entry['cache_id']}: {entry['size']} tokens, {entry['hits']} hits")
    
    # Retrieve a cached context
    print(f"\n🔍 Retrieving cache {cache_id_1}...")
    entry = cache.get(cache_id_1)
    if entry:
        print(f"✓ Retrieved! Context size: {len(entry.context)} chars")
        print(f"  Hit count: {entry.metadata.hit_count}")
    
    # Clear cache
    print("\n🧹 Clearing cache...")
    cleared = cache.clear_all()
    print(f"✓ Cleared {cleared} entries")


def example_duplicate_detection():
    """Demonstrate duplicate context detection."""
    print("\n" + "=" * 60)
    print("Duplicate Context Detection Example")
    print("=" * 60)
    
    cache = KVCache(memory_only=True)
    
    # Same context
    context = "This is a test context for caching." * 50
    
    print("\n📝 Creating first cache...")
    cache_id_1 = cache.create(context, source_ids=["source1"])
    print(f"✓ Cache ID: {cache_id_1}")
    
    print("\n📝 Creating second cache with SAME context...")
    cache_id_2 = cache.create(context, source_ids=["source2"])
    print(f"✓ Cache ID: {cache_id_2}")
    
    if cache_id_1 == cache_id_2:
        print("\n✨ Duplicate detection works! Same cache ID returned")
    else:
        print(f"\n⚠️  Different IDs: {cache_id_1} vs {cache_id_2}")


def example_persistent_cache():
    """Demonstrate disk-based persistent cache."""
    print("\n" + "=" * 60)
    print("Persistent Cache Example")
    print("=" * 60)
    
    # Use disk cache
    cache_dir = Path(".example_cache")
    cache = KVCache(cache_dir=cache_dir, memory_only=False)
    
    context = "Sample document content for persistent caching." * 100
    
    print(f"\n💾 Creating cache with disk storage...")
    print(f"   Cache directory: {cache_dir}")
    
    cache_id = cache.create(context, source_ids=["persistent_doc.pdf"])
    print(f"✓ Created cache: {cache_id}")
    
    # Stats
    stats = cache.get_stats()
    print(f"\n📊 Cache Stats:")
    print(f"  Entries: {stats['total_entries']}")
    print(f"  Disk storage: {cache_dir / f'{cache_id}.pkl'}")
    
    # Cleanup
    print(f"\n🧹 Cleaning up example cache...")
    import shutil
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    print("✓ Cleaned up")


def example_multi_source_cache():
    """Demonstrate caching multiple sources together."""
    print("\n" + "=" * 60)
    print("Multi-Source Cache Example")
    print("=" * 60)
    
    cache = KVCache(memory_only=True)
    
    # Multiple document contexts
    doc1 = "Document 1 content..." * 50
    doc2 = "Document 2 content..." * 50
    doc3 = "Document 3 content..." * 50
    
    # Combine into single context
    combined_context = f"{doc1}\n{doc2}\n{doc3}"
    
    # Create cache with source tracking
    source_ids = ["report.pdf", "slides.pptx", "notes.txt"]
    cache_id = cache.create(combined_context, source_ids=source_ids)
    
    print(f"\n📝 Created cache with {len(source_ids)} sources:")
    for source_id in source_ids:
        print(f"   - {source_id}")
    
    # Retrieve and show metadata
    entry = cache.get(cache_id)
    print(f"\n📊 Cache Metadata:")
    print(f"   Cache ID: {cache_id}")
    print(f"   Context size: {entry.metadata.context_size} tokens")
    print(f"   Sources: {', '.join(entry.metadata.source_ids)}")
    print(f"   Created: {entry.metadata.created_at}")


if __name__ == "__main__":
    print("\n" + "🚀 " * 20)
    print("KV-Cache for Cache-Augmented Generation (CAG)")
    print("🚀 " * 20)
    
    # Run examples
    example_basic_usage()
    example_duplicate_detection()
    example_multi_source_cache()
    example_persistent_cache()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
