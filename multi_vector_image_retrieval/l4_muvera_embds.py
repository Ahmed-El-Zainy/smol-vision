"""
MUVERA Compression and Two-Stage Retrieval System
Compares ColPali multi-vectors with MUVERA fixed-dimensional encodings
and demonstrates efficient two-stage retrieval
"""

import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Any

import numpy as np
from tqdm import tqdm
from qdrant_client import QdrantClient, models
from fastembed.postprocess.muvera import Muvera

from helper import (
    load_sample_image_embeddings,
    load_or_compute_query_embeddings,
    yield_muvera_embeddings,
    compare_search_methods,
)

# Configuration
LOAD_PRECOMPUTED = True
COLLECTION_NAME = "colpali-optimizations"
QDRANT_URL = "http://localhost:6333"
OUTPUT_DIR = Path("/mnt/user-data/outputs/muvera")

# MUVERA Configuration
MUVERA_DIM = 128  # ColPali token embedding dimensionality
MUVERA_K_SIM = 64  # Number of clusters
MUVERA_DIM_PROJ = 16  # Random projection dimension
MUVERA_R_REPS = 20  # Repetitions to concatenate
MUVERA_RANDOM_SEED = 42  # Reproducibility

# Derived values
MUVERA_FDE_SIZE = MUVERA_K_SIM * MUVERA_DIM_PROJ * MUVERA_R_REPS  # 20,480


def setup_logging() -> logging.Logger:
    """Configure logging with both file and console handlers"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(OUTPUT_DIR / "muvera_retrieval.log")
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def initialize_muvera(logger: logging.Logger) -> Muvera:
    """Initialize MUVERA compression model"""
    logger.info("=" * 80)
    logger.info("Initializing MUVERA")
    logger.info("=" * 80)
    
    try:
        muvera = Muvera(
            dim=MUVERA_DIM,
            k_sim=MUVERA_K_SIM,
            dim_proj=MUVERA_DIM_PROJ,
            r_reps=MUVERA_R_REPS,
            random_seed=MUVERA_RANDOM_SEED,
        )
        
        logger.info("MUVERA initialized successfully")
        logger.info(f"  Token dimension: {MUVERA_DIM}")
        logger.info(f"  Clusters (k_sim): {MUVERA_K_SIM}")
        logger.info(f"  Projection dimension: {MUVERA_DIM_PROJ}")
        logger.info(f"  Repetitions: {MUVERA_R_REPS}")
        logger.info(f"  Output FDE size: {MUVERA_FDE_SIZE}")
        logger.info(f"  Random seed: {MUVERA_RANDOM_SEED}")
        
        return muvera
        
    except Exception as e:
        logger.error(f"Failed to initialize MUVERA: {e}")
        raise


def load_and_process_documents(
    muvera: Muvera,
    logger: logging.Logger
) -> Any:
    """Load image embeddings and process with MUVERA"""
    logger.info("=" * 80)
    logger.info("Loading and Processing Document Embeddings")
    logger.info("=" * 80)
    
    try:
        # Load image embeddings
        logger.info(f"Load precomputed: {LOAD_PRECOMPUTED}")
        images_df = load_sample_image_embeddings(
            load_precomputed=LOAD_PRECOMPUTED,
        )
        
        logger.info(f"Loaded {len(images_df)} document pages")
        
        # Display sample
        logger.info("\nSample documents:")
        for idx in range(min(3, len(images_df))):
            logger.info(f"  {idx}: {images_df['image_path'].iloc[idx]}")
        
        # Process all documents with MUVERA
        logger.info("\nProcessing documents with MUVERA...")
        muvera_embeddings = []
        
        for _, row in tqdm(
            images_df.iterrows(),
            total=len(images_df),
            desc="MUVERA embeddings"
        ):
            muvera_fde = muvera.process_document(row["image_embedding"])
            muvera_embeddings.append(muvera_fde)
        
        # Add to dataframe
        images_df["muvera_embedding"] = muvera_embeddings
        
        # Log shape comparison
        original_shape = images_df['image_embedding'].iloc[0].shape
        muvera_shape = images_df['muvera_embedding'].iloc[0].shape
        
        logger.info(f"\nCompression Results:")
        logger.info(f"  Original shape: {original_shape}")
        logger.info(f"  MUVERA FDE shape: {muvera_shape}")
        
        # Calculate compression ratio
        original_size = np.prod(original_shape)
        muvera_size = np.prod(muvera_shape)
        compression_ratio = original_size / muvera_size
        
        logger.info(f"  Original size: {original_size:,} values")
        logger.info(f"  MUVERA size: {muvera_size:,} values")
        logger.info(f"  Compression ratio: {compression_ratio:.2f}x")
        
        return images_df
        
    except Exception as e:
        logger.error(f"Failed to load and process documents: {e}", exc_info=True)
        raise


def load_and_process_queries(
    muvera: Muvera,
    logger: logging.Logger
) -> Tuple[List[str], List[np.ndarray], List[np.ndarray]]:
    """Load query embeddings and process with MUVERA"""
    logger.info("=" * 80)
    logger.info("Loading and Processing Query Embeddings")
    logger.info("=" * 80)
    
    try:
        # Load query embeddings
        logger.info(f"Load precomputed: {LOAD_PRECOMPUTED}")
        queries_df = load_or_compute_query_embeddings(
            load_precomputed=LOAD_PRECOMPUTED,
        )
        
        # Extract queries and embeddings
        queries = queries_df["query"].tolist()
        query_embeddings = queries_df["query_embedding"].tolist()
        
        logger.info(f"Loaded {len(queries)} queries:")
        for i, query in enumerate(queries):
            logger.info(f"  Query {i + 1}: '{query}'")
        
        # Process queries with MUVERA
        logger.info("\nProcessing queries with MUVERA...")
        muvera_query_embeddings = []
        
        for qe in tqdm(query_embeddings, desc="MUVERA query embeddings"):
            qe_array = np.stack(qe)
            muvera_qe = muvera.process_query(qe_array)
            muvera_query_embeddings.append(muvera_qe)
        
        # Log shape comparison
        original_shape = np.stack(query_embeddings[0]).shape
        muvera_shape = muvera_query_embeddings[0].shape
        
        logger.info(f"\nQuery Compression Results:")
        logger.info(f"  Original query shape: {original_shape}")
        logger.info(f"  MUVERA query FDE shape: {muvera_shape}")
        
        return queries, query_embeddings, muvera_query_embeddings
        
    except Exception as e:
        logger.error(f"Failed to load and process queries: {e}", exc_info=True)
        raise


def create_muvera_collection(
    client: QdrantClient,
    logger: logging.Logger
) -> None:
    """Create Qdrant collection with ColPali and MUVERA vectors"""
    logger.info("=" * 80)
    logger.info("Creating Qdrant Collection")
    logger.info("=" * 80)
    
    try:
        # Delete existing collection
        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)
            logger.info(f"Deleted existing collection: {COLLECTION_NAME}")
        
        # Create collection with dual vectors
        client.create_collection(
            COLLECTION_NAME,
            vectors_config={
                # Original ColPali multivectors
                "colpali_original": models.VectorParams(
                    size=MUVERA_DIM,
                    distance=models.Distance.DOT,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM,
                    ),
                    hnsw_config=models.HnswConfigDiff(m=0),
                    on_disk=True,
                ),
                # MUVERA fixed-dimensional encodings
                "muvera_fde": models.VectorParams(
                    size=MUVERA_FDE_SIZE,
                    distance=models.Distance.DOT,
                    on_disk=True,
                    # No multivector config - single vector with HNSW
                ),
            },
        )
        
        logger.info(f"Created collection: {COLLECTION_NAME}")
        logger.info("\nVector configurations:")
        logger.info(f"  1. colpali_original:")
        logger.info(f"     - Size: {MUVERA_DIM}")
        logger.info(f"     - Type: Multi-vector (MaxSim)")
        logger.info(f"     - Distance: DOT")
        logger.info(f"  2. muvera_fde:")
        logger.info(f"     - Size: {MUVERA_FDE_SIZE}")
        logger.info(f"     - Type: Single vector (HNSW)")
        logger.info(f"     - Distance: DOT")
        
    except Exception as e:
        logger.error(f"Failed to create collection: {e}", exc_info=True)
        raise


def populate_muvera_collection(
    client: QdrantClient,
    muvera: Muvera,
    logger: logging.Logger
) -> int:
    """Populate collection with ColPali and MUVERA embeddings"""
    logger.info("=" * 80)
    logger.info("Populating Collection")
    logger.info("=" * 80)
    
    try:
        logger.info(f"Load precomputed: {LOAD_PRECOMPUTED}")
        logger.info("Streaming through embeddings and upserting...")
        
        count = 0
        
        # Stream through embeddings
        for i, (image_path, vectors) in enumerate(
            tqdm(
                yield_muvera_embeddings(
                    muvera=muvera,
                    load_precomputed=LOAD_PRECOMPUTED,
                ),
                desc="Processing and inserting documents",
            )
        ):
            client.upsert(
                COLLECTION_NAME,
                points=[
                    models.PointStruct(
                        id=i,
                        vector={
                            "colpali_original": vectors["colpali_original"],
                            "muvera_fde": vectors["muvera_fde"],
                        },
                        payload={
                            "image_path": image_path,
                        },
                    )
                ],
            )
            count += 1
        
        logger.info(f"\nInserted {count} documents into {COLLECTION_NAME}")
        
        # Wait for indexing
        logger.info("Waiting for collection to finish indexing...")
        time.sleep(5.0)
        
        while True:
            collection_info = client.get_collection(COLLECTION_NAME)
            if collection_info.status == models.CollectionStatus.GREEN:
                break
            logger.info("  Still indexing...")
            time.sleep(5.0)
        
        logger.info("Collection has indexed all data points")
        
        return count
        
    except Exception as e:
        logger.error(f"Failed to populate collection: {e}", exc_info=True)
        raise


def search_colpali(
    client: QdrantClient,
    query_embedding: np.ndarray,
    limit: int = 5
) -> Tuple[List[models.ScoredPoint], float]:
    """Search using original ColPali multivectors"""
    start = time.time()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        using="colpali_original",
        limit=limit,
        with_payload=True,
    )
    search_time = time.time() - start
    return results.points, search_time


def search_muvera(
    client: QdrantClient,
    query_embedding: np.ndarray,
    limit: int = 5
) -> Tuple[List[models.ScoredPoint], float]:
    """Search using MUVERA compressed vectors"""
    start = time.time()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        using="muvera_fde",
        limit=limit,
        with_payload=True,
    )
    search_time = time.time() - start
    return results.points, search_time


def two_stage_retrieval(
    client: QdrantClient,
    query_colpali: np.ndarray,
    query_muvera: np.ndarray,
    limit: int = 5
) -> Tuple[List[models.ScoredPoint], float]:
    """
    Two-stage retrieval using prefetch:
    1. Fast MUVERA search for candidates
    2. Rerank with ColPali for accuracy
    """
    start = time.time()
    
    # Single API call with prefetch mechanism
    final_results = client.query_points(
        prefetch=[
            models.Prefetch(
                query=query_muvera,
                using="muvera_fde",
                limit=limit * 10,  # Ten times more candidates
            )
        ],
        collection_name=COLLECTION_NAME,
        query=query_colpali,
        using="colpali_original",
        limit=limit,
        with_payload=True,
    )
    
    total_time = time.time() - start
    
    return final_results.points, total_time


def run_muvera_comparisons(
    client: QdrantClient,
    queries: List[str],
    query_embeddings: List[np.ndarray],
    muvera_query_embeddings: List[np.ndarray],
    logger: logging.Logger
) -> List[Dict]:
    """Run ColPali vs MUVERA comparisons for all queries"""
    logger.info("=" * 80)
    logger.info("Running ColPali vs MUVERA Comparisons")
    logger.info("=" * 80)
    
    results = []
    
    for idx, query in enumerate(queries):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Query {idx + 1}: '{query}'")
        logger.info('=' * 80)
        
        try:
            result = compare_search_methods(
                baseline_search_fn=lambda: search_colpali(
                    client, query_embeddings[idx], limit=5
                ),
                comparison_search_fn=lambda: search_muvera(
                    client, muvera_query_embeddings[idx], limit=5
                ),
                baseline_name="ColPali",
                comparison_name="MUVERA",
                query_text=query,
                limit=5,
                n_runs=10,
            )
            
            results.append(result)
            
            # Log individual results
            logger.info(f"\nResults for '{query}':")
            logger.info(f"  Average speedup: {result['avg_speedup']:.1f}x")
            logger.info(f"  Median speedup: {result['median_speedup']:.1f}x")
            logger.info(f"  Precision@5: {result['precision']:.1%}")
            
        except Exception as e:
            logger.error(f"Failed comparison for query {idx + 1}: {e}", exc_info=True)
    
    # Calculate and log averages
    if results:
        avg_speedup = np.mean([r["avg_speedup"] for r in results])
        median_speedup = np.mean([r["median_speedup"] for r in results])
        avg_precision = np.mean([r["precision"] for r in results])
        
        logger.info("\n" + "=" * 80)
        logger.info("AVERAGE PERFORMANCE (ColPali vs MUVERA)")
        logger.info("=" * 80)
        logger.info(f"Average speedup (mean): {avg_speedup:.1f}x faster")
        logger.info(f"Average speedup (median): {median_speedup:.1f}x faster")
        logger.info(f"Average precision@5: {avg_precision:.1%}")
        
        # Save summary to file
        summary_path = OUTPUT_DIR / "muvera_comparison_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("MUVERA vs ColPali Comparison Summary\n")
            f.write("=" * 60 + "\n\n")
            for idx, (query, result) in enumerate(zip(queries, results)):
                f.write(f"Query {idx + 1}: {query}\n")
                f.write(f"  Speedup: {result['avg_speedup']:.1f}x\n")
                f.write(f"  Precision: {result['precision']:.1%}\n\n")
            f.write(f"\nOverall Averages:\n")
            f.write(f"  Average speedup: {avg_speedup:.1f}x\n")
            f.write(f"  Median speedup: {median_speedup:.1f}x\n")
            f.write(f"  Average precision: {avg_precision:.1%}\n")
        
        logger.info(f"\nSaved summary to: {summary_path}")
    
    return results


def run_two_stage_comparisons(
    client: QdrantClient,
    queries: List[str],
    query_embeddings: List[np.ndarray],
    muvera_query_embeddings: List[np.ndarray],
    logger: logging.Logger
) -> List[Dict]:
    """Run two-stage retrieval comparisons for all queries"""
    logger.info("=" * 80)
    logger.info("Running Two-Stage Retrieval Comparisons")
    logger.info("=" * 80)
    
    results = []
    
    for idx, query in enumerate(queries):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Query {idx + 1}: '{query}'")
        logger.info('=' * 80)
        
        try:
            result = compare_search_methods(
                baseline_search_fn=lambda: search_colpali(
                    client, query_embeddings[idx], limit=5
                ),
                comparison_search_fn=lambda: two_stage_retrieval(
                    client,
                    query_embeddings[idx],
                    muvera_query_embeddings[idx],
                    limit=5,
                ),
                baseline_name="ColPali",
                comparison_name="Two-stage",
                query_text=query,
                limit=5,
                n_runs=10,
            )
            
            results.append(result)
            
            # Log individual results
            logger.info(f"\nResults for '{query}':")
            logger.info(f"  Two-stage time: {result['comparison_avg_time'] * 1000:.2f}ms")
            logger.info(f"  Precision@5: {result['precision']:.1%}")
            
        except Exception as e:
            logger.error(f"Failed two-stage comparison for query {idx + 1}: {e}", exc_info=True)
    
    # Calculate and log averages
    if results:
        avg_two_stage_time = np.mean([r["comparison_avg_time"] for r in results])
        avg_precision = np.mean([r["precision"] for r in results])
        
        logger.info("\n" + "=" * 80)
        logger.info("TWO-STAGE RETRIEVAL SUMMARY")
        logger.info("=" * 80)
        logger.info(f"\nAverage two-stage time: {avg_two_stage_time * 1000:.2f}ms")
        for idx, result in enumerate(results):
            logger.info(f"  Query {idx + 1}: {result['comparison_avg_time'] * 1000:.2f}ms")
        logger.info(f"\nAverage precision@5 vs ColPali: {avg_precision:.1%}")
        
        # Save summary to file
        summary_path = OUTPUT_DIR / "two_stage_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("Two-Stage Retrieval Summary\n")
            f.write("=" * 60 + "\n\n")
            for idx, (query, result) in enumerate(zip(queries, results)):
                f.write(f"Query {idx + 1}: {query}\n")
                f.write(f"  Time: {result['comparison_avg_time'] * 1000:.2f}ms\n")
                f.write(f"  Precision: {result['precision']:.1%}\n\n")
            f.write(f"\nOverall Averages:\n")
            f.write(f"  Average time: {avg_two_stage_time * 1000:.2f}ms\n")
            f.write(f"  Average precision: {avg_precision:.1%}\n")
        
        logger.info(f"\nSaved summary to: {summary_path}")
    
    return results


def main():
    """Main execution function"""
    logger = setup_logging()
    
    logger.info("=" * 80)
    logger.info("MUVERA Compression and Two-Stage Retrieval System")
    logger.info("=" * 80)
    
    try:
        # Initialize MUVERA
        muvera = initialize_muvera(logger)
        
        # Load and process documents
        images_df = load_and_process_documents(muvera, logger)
        
        # Load and process queries
        queries, query_embeddings, muvera_query_embeddings = load_and_process_queries(
            muvera, logger
        )
        
        # Connect to Qdrant
        logger.info(f"\nConnecting to Qdrant at: {QDRANT_URL}")
        client = QdrantClient(QDRANT_URL)
        logger.info("Successfully connected to Qdrant")
        
        # Create collection
        create_muvera_collection(client, logger)
        
        # Populate collection
        doc_count = populate_muvera_collection(client, muvera, logger)
        
        # Run MUVERA comparisons
        muvera_results = run_muvera_comparisons(
            client, queries, query_embeddings, muvera_query_embeddings, logger
        )
        
        # Run two-stage comparisons
        two_stage_results = run_two_stage_comparisons(
            client, queries, query_embeddings, muvera_query_embeddings, logger
        )
        
        logger.info("=" * 80)
        logger.info("Execution completed successfully!")
        logger.info(f"Processed {doc_count} documents")
        logger.info(f"Ran {len(queries)} test queries")
        logger.info(f"All outputs saved to: {OUTPUT_DIR}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Fatal error during execution: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()