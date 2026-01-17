# Multi-Vector Image Retrieval Optimization Challenge

## Goal

Optimize multi-vector search for the "Attention is All You Need" research paper using **Scalar Quantization** with optional **HNSW** for speed.

## What's Pre-Built

- **Baseline ColPali Search**: Already indexed 5 pages from the paper with pre-computed embeddings
- **Helper Module**: `baseline_helper.py` handles all baseline setup
- **Baseline Results**: You can see search results immediately after running Cell 2

## Your Task

**Implement ONE optimization approach** in Cell 4 (the only cell you need to fill out):

### Option 1: Scalar Quantization Only (Recommended - Start Here)
- **Goal**: Reduce memory by 75% (float32 → int8)
- **HNSW**: Disabled (exact search with m=0)
- **Trade-off**: No accuracy loss, similar speed
- **Complexity**: Simple - Qdrant handles everything internally
- **When to use**: Memory-constrained environments
- **Collection name**: `"optimized"`

### Option 2: Scalar Quantization + HNSW (Advanced)
- **Goal**: Reduce memory by 75% AND speed up search
- **HNSW**: Enabled (approximate search with m=16, ef_construct=100)
- **Trade-off**: Slight accuracy loss for speed gains
- **Complexity**: Same as Option 1, just one parameter change
- **When to use**: Large-scale production systems (1000+ documents)
- **Collection name**: `"optimized_hnsw"`
- **Note**: 5 images too small to show HNSW benefits!

## Dataset

**5 pages from "Attention is All You Need" paper** (pre-indexed):
- `page-0.png` - Title page
- `page-1.png` - Abstract
- `page-3.png` - Architecture diagram
- `page-4.png` - Attention mechanism
- `page-8.png` - Results

**Pre-computed embeddings**: `attention_paper/attention-embeddings-colsmol.parquet`

## Example Query

The baseline already ran: `"transformer architecture diagram"`

**Expected**: Should match `page-8.png` (results section with architecture diagram)

## Success Criteria

After implementing your optimization:

1. **Search still works**: Returns relevant pages
2. **Measure improvement**:
   - **Option 1**: ~75% memory reduction, similar speed, perfect accuracy
   - **Option 2**: ~75% memory reduction, slight accuracy trade-off
3. **Compare accuracy**: Check if top result matches baseline

## Files to Attach to Jupyter AI Prompts

When asking Jupyter AI for help, attach these 3 files:

1. **`project.ipynb`** - Your notebook with progress
2. **`spec.md`** - This file (project specification)
3. **`docs.md`** - API documentation for Scalar Quantization

## Optimization Details

### Scalar Quantization (Both Options)

**How it works**:
- Converts float32 (4 bytes) → int8 (1 byte) per dimension
- Qdrant learns min/max values and maps linearly to 0-255
- Enabled at collection creation time
- Send original float32 embeddings - Qdrant quantizes internally

**Expected outcomes**:
- Memory: ~2.5 MB → ~0.6 MB (75% reduction)
- Speed: Similar to baseline (Option 1) or potentially faster (Option 2 on large datasets)
- Accuracy: Perfect (Option 1) or very high (Option 2, ~99% precision)

### HNSW (Option 2 Only)

**How it works**:
- Builds a hierarchical graph for approximate nearest neighbor search
- Trades accuracy for speed on large datasets
- Controlled by `m` (graph connectivity) and `ef_construct` (build quality)
- Disable with `m=0` for exact search (Option 1)

**Expected outcomes**:
- Speed: 10-100x faster on 10,000+ vectors (minimal impact on 5 vectors!)
- Accuracy: Slight trade-off (typically 95-99% recall)
- Best for: Production systems with 1000+ documents

## Key Learning Points

**Option 1 (Scalar Only)**:
- Memory optimization without accuracy loss
- Exact search maintained
- Simple, reliable, production-ready

**Option 2 (Scalar + HNSW)**:
- Stacking optimizations (memory + speed)
- Understanding HNSW requires dataset scale
- When to trade accuracy for speed

## Time Estimate

**10 minutes total**:
- 2 minutes: Run baseline (Cell 2)
- 5 minutes: Copy example prompt to Jupyter AI, generate code
- 3 minutes: Run optimization, compare results
- Optional: Try the other option and compare!

## Comparison Function

After implementing your optimization, use the helper function:

```python
from baseline_helper import print_comparison

print_comparison(
    optimized_results,      # Your search results
    optimized_metrics,      # Your metrics dict
    baseline_metrics,       # From baseline.get_baseline_metrics()
    optimization_type="Scalar Quantization"  # or "Scalar + HNSW"
)
```

This will show a formatted comparison of memory, speed, and accuracy.

## Production Tips

1. **Start simple**: Begin with Option 1 (Scalar only)
2. **Scale up**: Add HNSW (Option 2) when reaching 1000+ documents
3. **Monitor metrics**: Use print_comparison() to track improvements
4. **Tune HNSW**: Increase `m` and `ef_construct` for better accuracy if needed
