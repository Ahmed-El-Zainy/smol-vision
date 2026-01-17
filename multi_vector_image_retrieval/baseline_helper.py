"""
Helper module for ColPali Multi-Vector Optimization Project
Handles all baseline setup, indexing, and search functionality
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import torch
import numpy as np
import time
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, MultiVectorConfig,
    MultiVectorComparator, PointStruct, HnswConfigDiff
)

class BaselineSetup:
    """Manages baseline ColPali search setup and benchmarking"""

    def __init__(self):
        self.client = QdrantClient(":memory:")
        self.embeddings_df = None
        self.baseline_results = None
        self.baseline_search_time = None
        self.total_vectors = 0
        self.vector_dim = 128
        self.baseline_memory_mb = 0

    def load_embeddings(self, parquet_path="attention_paper/attention-embeddings-colsmol.parquet", available_pages=None):
        """Load pre-computed ColPali embeddings from parquet

        Args:
            parquet_path: Path to embeddings parquet file
            available_pages: List of page indices to use (default: [0, 1, 2, 3, 4])
        """
        print("📦 Loading pre-computed embeddings...")
        self.embeddings_df = pd.read_parquet(parquet_path)

        # Use first 5 pages from the paper by default
        if available_pages is None:
            available_pages = [0, 1, 2, 3, 4]

        self.embeddings_df = self.embeddings_df[
            self.embeddings_df.index.isin(available_pages)
        ].copy()

        # Calculate memory metrics
        self.total_vectors = sum(len(row['image_embedding']) for _, row in self.embeddings_df.iterrows())
        self.vector_dim = len(self.embeddings_df.iloc[0]['image_embedding'][0])
        self.baseline_memory_mb = (self.total_vectors * self.vector_dim * 4) / (1024 * 1024)

        print(f"✅ Loaded {len(self.embeddings_df)} images")
        print(f"   Total vectors: {self.total_vectors}")
        print(f"   Vector dimension: {self.vector_dim}")
        print(f"   Memory (float32): {self.baseline_memory_mb:.2f} MB")

        return self.embeddings_df

    def create_baseline_collection(self):
        """Create and populate baseline Qdrant collection"""
        print("\n🗄️  Creating baseline collection...")

        # Create collection
        self.client.create_collection(
            collection_name="baseline",
            vectors_config={
                "colpali": VectorParams(
                    size=self.vector_dim,
                    distance=Distance.DOT,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM
                    ),
                    hnsw_config=HnswConfigDiff(m=0)
                )
            }
        )

        # Index embeddings
        for idx, row in self.embeddings_df.iterrows():
            self.client.upsert(
                collection_name="baseline",
                points=[
                    PointStruct(
                        id=idx,
                        vector={"colpali": row['image_embedding']},
                        payload={
                            "image_path": row['file_path'],
                            "page_number": idx
                        }
                    )
                ]
            )

        print(f"✅ Indexed {len(self.embeddings_df)} images")

    def load_model(self, model_name="vidore/colSmol-256M"):
        """Load ColPali model for query encoding"""
        print("\n🤖 Loading ColPali model...")
        from colpali_engine.models import ColIdefics3Processor, ColIdefics3

        self.processor = ColIdefics3Processor.from_pretrained(model_name)
        self.model = ColIdefics3.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        print("✅ Model loaded!")

        return self.processor, self.model

    def search_baseline(self, query="transformer architecture diagram", limit=3):
        """Run baseline search and return results with metrics"""
        print(f"\n🔍 Searching: '{query}'")

        # Encode query
        batch_queries = self.processor.process_queries([query]).to(self.model.device)
        with torch.no_grad():
            query_embeddings = self.model(**batch_queries).to(dtype=torch.float32)

        # Search
        start_time = time.time()
        self.baseline_results = self.client.query_points(
            collection_name="baseline",
            query=query_embeddings[0].cpu().numpy(),
            using="colpali",
            limit=limit,
            with_payload=True
        )
        self.baseline_search_time = time.time() - start_time

        # Display results
        print("\n📊 BASELINE RESULTS:")
        for i, hit in enumerate(self.baseline_results.points, 1):
            print(f"  {i}. Score: {hit.score:.4f} | {hit.payload['image_path']}")

        print(f"\n💾 Memory: {self.baseline_memory_mb:.2f} MB")
        print(f"⏱️  Search Time: {self.baseline_search_time*1000:.2f}ms")
        print(f"📈 Total Vectors: {self.total_vectors}")

        return query_embeddings[0].cpu().numpy()

    def get_baseline_metrics(self):
        """Return baseline metrics for comparison"""
        return {
            'top_score': self.baseline_results.points[0].score,
            'top_page': self.baseline_results.points[0].payload['image_path'],
            'search_time': self.baseline_search_time,
            'memory_mb': self.baseline_memory_mb,
            'total_vectors': self.total_vectors,
            'vector_dim': self.vector_dim
        }

    def setup_all(self, query="transformer architecture diagram", available_pages=None):
        """Complete setup: load embeddings, create collection, load model, run search

        Args:
            query: Search query text (default: "transformer architecture diagram")
            available_pages: List of page indices to use (default: [0, 1, 2, 3, 4])
        """
        self.load_embeddings(available_pages=available_pages)
        self.create_baseline_collection()
        self.load_model()
        query_embedding = self.search_baseline(query=query)

        return query_embedding, self.get_baseline_metrics()

def print_comparison(optimized_results, optimized_metrics, baseline_metrics, optimization_type="Optimized"):
    """Print formatted comparison between baseline and optimized results"""
    print(f"\n{'='*60}")
    print(f"📊 COMPARISON: BASELINE vs {optimization_type.upper()}")
    print(f"{'='*60}")

    # Memory comparison
    if 'memory_mb' in optimized_metrics:
        memory_reduction = ((baseline_metrics['memory_mb'] - optimized_metrics['memory_mb']) /
                           baseline_metrics['memory_mb']) * 100
        print(f"\n💾 MEMORY:")
        print(f"   Baseline:  {baseline_metrics['memory_mb']:.2f} MB")
        print(f"   Optimized: {optimized_metrics['memory_mb']:.2f} MB")
        print(f"   Reduction: {memory_reduction:.1f}% 📉")

    # Vector comparison
    if 'total_vectors' in optimized_metrics:
        vector_reduction = ((baseline_metrics['total_vectors'] - optimized_metrics['total_vectors']) /
                           baseline_metrics['total_vectors']) * 100
        print(f"\n📈 VECTORS:")
        print(f"   Baseline:  {baseline_metrics['total_vectors']}")
        print(f"   Optimized: {optimized_metrics['total_vectors']}")
        print(f"   Reduction: {vector_reduction:.1f}% 📉")

    # Speed comparison
    search_speedup = ((baseline_metrics['search_time'] - optimized_metrics['search_time']) /
                      baseline_metrics['search_time']) * 100
    print(f"\n⏱️  SEARCH TIME:")
    print(f"   Baseline:  {baseline_metrics['search_time']*1000:.2f}ms")
    print(f"   Optimized: {optimized_metrics['search_time']*1000:.2f}ms")
    if search_speedup > 0:
        print(f"   Speedup:   {search_speedup:.1f}% faster ⚡")
    else:
        print(f"   Change:    {abs(search_speedup):.1f}% slower")

    # Accuracy comparison
    top_match = optimized_results.points[0].payload['image_path'] == baseline_metrics['top_page']
    score_diff = abs(optimized_results.points[0].score - baseline_metrics['top_score'])

    print(f"\n🎯 ACCURACY:")
    print(f"   Top page matches: {'✅ YES' if top_match else '❌ NO'}")
    print(f"   Score difference: {score_diff:.4f}")

    print(f"\n{'='*60}")
