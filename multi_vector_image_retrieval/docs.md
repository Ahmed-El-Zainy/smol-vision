# Qdrant Multi-Vector Optimization API Reference

This document covers the Qdrant Python client APIs for optimizing ColPali multi-vector search using scalar quantization.

## Scalar Quantization for Multi-Vector Search

Scalar quantization compresses float32 vectors (4 bytes per dimension) to int8 (1 byte per dimension), achieving 4x memory reduction with minimal accuracy loss.

### Creating a Collection with Scalar Quantization

```python
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, MultiVectorConfig, MultiVectorComparator,
    HnswConfigDiff, ScalarQuantization, ScalarQuantizationConfig, ScalarType
)

client = QdrantClient(":memory:")

client.create_collection(
    collection_name="optimized",
    vectors_config={
        "colpali": VectorParams(
            size=128,  # vector dimension
            distance=Distance.DOT,
            multivector_config=MultiVectorConfig(
                comparator=MultiVectorComparator.MAX_SIM
            ),
            hnsw_config=HnswConfigDiff(m=0),  # Disable HNSW for exact search
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8  # 4x compression
                )
            )
        )
    }
)
```

### Enabling HNSW for Speed (Optional)

For large datasets (1000+ documents), enable HNSW approximate search:

```python
client.create_collection(
    collection_name="optimized_hnsw",
    vectors_config={
        "colpali": VectorParams(
            size=128,
            distance=Distance.DOT,
            multivector_config=MultiVectorConfig(
                comparator=MultiVectorComparator.MAX_SIM
            ),
            hnsw_config=HnswConfigDiff(
                m=16,           # Graph connectivity (higher = better accuracy, more memory)
                ef_construct=100  # Build quality (higher = better accuracy, slower indexing)
            ),
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8
                )
            )
        )
    }
)
```

**Key parameters:**
- `m=0`: Disables HNSW (exact search) - recommended for small datasets
- `m=16`: Enables HNSW (approximate search) - recommended for 1000+ documents
- `ef_construct=100`: Controls index build quality (higher = better accuracy)

### Indexing with Scalar Quantization

```python
from qdrant_client.models import PointStruct

# Insert embeddings - Qdrant handles quantization internally
client.upsert(
    collection_name="optimized",
    points=[
        PointStruct(
            id=0,
            vector={"colpali": original_float32_embedding},  # Send original vectors
            payload={"image_path": "page-0.png"}
        )
    ]
)
```

**Important**: Always send original float32 embeddings. Qdrant quantizes them internally.

### Searching with Scalar Quantization

```python
# Encode query using ColPali
import torch
from colpali_engine.models import ColIdefics3Processor, ColIdefics3

processor = ColIdefics3Processor.from_pretrained("vidore/colSmol-256M")
model = ColIdefics3.from_pretrained(
    "vidore/colSmol-256M",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager"
)

batch_queries = processor.process_queries([query_text]).to(model.device)
with torch.no_grad():
    query_embedding = model(**batch_queries).to(dtype=torch.float32)[0].cpu().numpy()

# Search - send original float32 query embedding
results = client.query_points(
    collection_name="optimized",
    query=query_embedding,
    using="colpali",
    limit=3,
    with_payload=True
)
```

## Measuring Performance Metrics

### Memory Usage

```python
# Scalar Quantization: 75% reduction
baseline_memory_mb = (total_vectors * 128 * 4) / (1024 * 1024)  # float32
quantized_memory_mb = (total_vectors * 128 * 1) / (1024 * 1024)  # int8

print(f"Baseline: {baseline_memory_mb:.2f} MB")
print(f"Quantized: {quantized_memory_mb:.2f} MB")
print(f"Reduction: {((baseline_memory_mb - quantized_memory_mb) / baseline_memory_mb) * 100:.1f}%")
```

### Search Time

```python
import time

start_time = time.time()
results = client.query_points(...)
search_time = time.time() - start_time

print(f"Search time: {search_time*1000:.2f}ms")
```

### Accuracy (Precision)

```python
# Compare top results between baseline and optimized
baseline_pages = {hit.payload['image_path'] for hit in baseline_results.points}
optimized_pages = {hit.payload['image_path'] for hit in optimized_results.points}

matches = len(baseline_pages & optimized_pages)
precision = matches / len(baseline_pages)
print(f"Precision: {precision*100:.1f}%")
```

## Trade-offs

### Scalar Quantization (HNSW Disabled)
- **Memory**: 75% reduction (float32 → int8)
- **Speed**: Similar to baseline (still brute-force MaxSim)
- **Accuracy**: Very high (~99-100% precision)
- **Best for**: Memory-constrained environments, small datasets

### Scalar Quantization + HNSW (HNSW Enabled)
- **Memory**: 75% reduction (same as above)
- **Speed**: 10-100x faster on large datasets (minimal impact on small datasets)
- **Accuracy**: High (~95-99% precision)
- **Best for**: Large-scale production systems (1000+ documents)

## Helper Functions Available

The `baseline_helper.py` module provides:

```python
from baseline_helper import BaselineSetup, print_comparison

# Access baseline results
baseline = BaselineSetup()
query_embedding, baseline_metrics = baseline.setup_all()

# Compare optimization results
print_comparison(
    optimized_results,
    optimized_metrics,
    baseline_metrics,
    optimization_type="Scalar Quantization"  # or "Scalar + HNSW"
)
```

## Common Patterns

### Pattern 1: Memory Optimization Only

Use when you need to reduce memory usage but maintain perfect accuracy:

```python
# Disable HNSW for exact search
hnsw_config=HnswConfigDiff(m=0)
```

### Pattern 2: Memory + Speed Optimization

Use when you have 1000+ documents and can accept slight accuracy trade-off:

```python
# Enable HNSW for approximate search
hnsw_config=HnswConfigDiff(m=16, ef_construct=100)
```

### Pattern 3: Tuning HNSW for Better Accuracy

If approximate search accuracy is too low:

```python
# Increase parameters for better accuracy (costs more memory/time)
hnsw_config=HnswConfigDiff(
    m=32,             # Higher connectivity
    ef_construct=200  # Better build quality
)
```

## Production Recommendations

1. **Start simple**: Begin with Scalar Quantization only (`m=0`)
2. **Measure baseline**: Track memory, speed, and accuracy
3. **Scale gradually**: Add HNSW when reaching 1000+ documents
4. **Monitor metrics**: Use print_comparison() after changes
5. **Tune as needed**: Adjust `m` and `ef_construct` based on requirements

## Example: Complete Implementation

```python
import time
import torch
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, MultiVectorConfig, MultiVectorComparator,
    PointStruct, HnswConfigDiff, ScalarQuantization,
    ScalarQuantizationConfig, ScalarType
)
from baseline_helper import print_comparison

# Step 1: Create optimized client
optimized_client = QdrantClient(":memory:")

# Step 2: Create collection with scalar quantization
optimized_client.create_collection(
    collection_name="optimized",
    vectors_config={
        "colpali": VectorParams(
            size=128,
            distance=Distance.DOT,
            multivector_config=MultiVectorConfig(
                comparator=MultiVectorComparator.MAX_SIM
            ),
            hnsw_config=HnswConfigDiff(m=0),  # Exact search
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(type=ScalarType.INT8)
            )
        )
    }
)

# Step 3: Index all images with original float32 embeddings
for idx, row in baseline.embeddings_df.iterrows():
    optimized_client.upsert(
        collection_name="optimized",
        points=[
            PointStruct(
                id=idx,
                vector={"colpali": row['image_embedding']},
                payload={"image_path": row['file_path'], "page_number": idx}
            )
        ]
    )

# Step 4: Run query
query_text = "transformer architecture diagram"
batch_queries = baseline.processor.process_queries([query_text]).to(baseline.model.device)
with torch.no_grad():
    query_embeddings = baseline.model(**batch_queries).to(dtype=torch.float32)

# Step 5: Search and measure time
start_time = time.time()
optimized_results = optimized_client.query_points(
    collection_name="optimized",
    query=query_embeddings[0].cpu().numpy(),
    using="colpali",
    limit=3,
    with_payload=True
)
optimized_search_time = time.time() - start_time

# Step 6: Calculate metrics
optimized_memory_mb = (baseline.total_vectors * baseline.vector_dim * 1) / (1024 * 1024)

optimized_metrics = {
    'top_score': optimized_results.points[0].score,
    'top_page': optimized_results.points[0].payload['image_path'],
    'search_time': optimized_search_time,
    'memory_mb': optimized_memory_mb,
    'total_vectors': baseline.total_vectors,
    'vector_dim': baseline.vector_dim
}

# Step 7: Compare with baseline
print_comparison(optimized_results, optimized_metrics, baseline_metrics,
                optimization_type="Scalar Quantization")
```
