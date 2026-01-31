"""
ColPali Optimization Comparison System
Compares various compression and pooling techniques for multi-vector embeddings
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from qdrant_client import QdrantClient, models
from colpali_engine.models import ColPaliProcessor
from colpali_engine.compression.token_pooling import HierarchicalTokenPooler

from helper import (
    load_sample_image_embeddings,
    load_or_compute_query_embeddings,
    yield_optimized_embeddings,
    compare_optimization_methods,
)

# Configuration
LOAD_PRECOMPUTED = True
MODEL_NAME = "vidore/colpali-v1.3"
COLLECTION_NAME = "colpali-optimizations"
QDRANT_URL = "http://localhost:6333"
OUTPUT_DIR = Path("/mnt/user-data/outputs/colpali_optimization")

# Model parameters
PATCH_SIZE = 32  # Default ColPali patch size
MODEL_DIM = 128  # Default ColPali embedding size

# Documents to process
ALLOWED_DOCS = (
    "AI4E_W1", "AI4E_W2", "DLS_C4W4", "GenAI4E_W1",
    "MLS_C2_W1", "MLS_C3_W2", "MLS_C2_W3", "DLS_C3_W1",
    "DLS_C1_W1", "MLS_C3_W3", "MLS_C1_W1",
)

# Optimization methods to compare
VECTOR_NAMES = [
    "original",  # Baseline
    "scalar_quantized",
    "binary_quantized",
    "row_pooled",
    "column_pooled",
    "hierarchical_2x",
    "hierarchical_4x",
]


def setup_logging() -> logging.Logger:
    """Configure logging with both file and console handlers"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(OUTPUT_DIR / "colpali_optimization.log")
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


def load_processor(logger: logging.Logger) -> ColPaliProcessor:
    """Load ColPali processor"""
    logger.info("=" * 80)
    logger.info("Loading ColPali Processor")
    logger.info("=" * 80)
    
    try:
        logger.info(f"Model: {MODEL_NAME}")
        processor = ColPaliProcessor.from_pretrained(MODEL_NAME)
        logger.info("Processor loaded successfully")
        return processor
        
    except Exception as e:
        logger.error(f"Failed to load processor: {e}")
        raise


def load_image_embeddings(logger: logging.Logger) -> Any:
    """Load sample image embeddings"""
    logger.info("=" * 80)
    logger.info("Loading Image Embeddings")
    logger.info("=" * 80)
    
    try:
        logger.info(f"Load precomputed: {LOAD_PRECOMPUTED}")
        images_df = load_sample_image_embeddings(
            load_precomputed=LOAD_PRECOMPUTED,
        )
        
        logger.info(f"Loaded {len(images_df)} image embeddings")
        
        # Add shape information
        images_df["original_shape"] = images_df["image_embedding"].apply(
            lambda x: x.shape
        )
        
        logger.info("\nSample entries:")
        sample_df = images_df.sample(n=min(3, len(images_df)), random_state=42)
        for idx, row in sample_df.iterrows():
            logger.info(f"  {idx}: {row['image_path']}, shape: {row['original_shape']}")
        
        return images_df
        
    except Exception as e:
        logger.error(f"Failed to load image embeddings: {e}", exc_info=True)
        raise


def visualize_sample_images(images_df: Any, logger: logging.Logger) -> None:
    """Visualize sample images from the dataset"""
    logger.info("=" * 80)
    logger.info("Visualizing Sample Images")
    logger.info("=" * 80)
    
    try:
        sample_df = images_df.sample(n=min(3, len(images_df)), random_state=42)
        
        fig, axarr = plt.subplots(3, 1, figsize=(9, 16))
        
        for i, (_, row) in enumerate(sample_df.iterrows()):
            image = Image.open(row["image_path"])
            axarr[i].imshow(image.convert("RGB"))
            axarr[i].set_title(f"Image: {row['image_path']}")
            axarr[i].axis('off')
        
        plt.tight_layout()
        output_path = OUTPUT_DIR / "sample_images.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved sample images to: {output_path}")
        
    except Exception as e:
        logger.warning(f"Could not visualize sample images: {e}")


def demonstrate_bucketing(logger: logging.Logger) -> None:
    """Demonstrate quantization bucketing concept"""
    logger.info("=" * 80)
    logger.info("Demonstrating Quantization Bucketing")
    logger.info("=" * 80)
    
    # Perform bucketing
    min_value, max_value = -0.8, 1.2
    value_space = np.linspace(min_value, max_value, 256)
    
    logger.info(f"Value range: [{min_value}, {max_value}]")
    logger.info(f"Number of buckets: 256")
    
    # Find which bucket index the target value belongs to
    target_value = 0.2
    bucket_index = np.argmin(np.abs(value_space - target_value))
    logger.info(f"Target value {target_value} → bucket index {bucket_index}")
    
    # Demonstrate binary quantization
    simple_vector = 2 * np.random.random(16) - 1.0
    binary_vector = (simple_vector > 0).astype(int)
    logger.info(f"\nBinary quantization example:")
    logger.info(f"  Original: {simple_vector[:5]}... (truncated)")
    logger.info(f"  Binary:   {binary_vector}")


def demonstrate_embedding_structure(
    images_df: Any,
    processor: ColPaliProcessor,
    logger: logging.Logger
) -> None:
    """Demonstrate how embeddings are structured"""
    logger.info("=" * 80)
    logger.info("Demonstrating Embedding Structure")
    logger.info("=" * 80)
    
    try:
        # Load first image
        image = Image.open(images_df["image_path"].iloc[0])
        logger.info(f"Image: {images_df['image_path'].iloc[0]}")
        
        # Process image
        batch_images = processor.process_images([image])
        decoded = processor.decode(batch_images.input_ids[0])
        logger.info(f"Decoded tokens (first 100 chars): {decoded[:100]}...")
        
        # Extract image embeddings
        image_mask = processor.get_image_mask(batch_images)[0]
        image_embedding = images_df["image_embedding"].iloc[0][image_mask]
        logger.info(f"Image embedding shape: {image_embedding.shape}")
        
    except Exception as e:
        logger.warning(f"Could not demonstrate embedding structure: {e}")


def embeddings_grid(image_embeddings: np.ndarray) -> np.ndarray:
    """
    Reshape the image embeddings so we have a grid of patches
    and their corresponding embeddings.
    """
    return image_embeddings.reshape((PATCH_SIZE, PATCH_SIZE, MODEL_DIM))


def row_mean_pooling(grid_embeddings: np.ndarray) -> np.ndarray:
    """Apply row-wise mean pooling"""
    return grid_embeddings.mean(axis=1)


def column_mean_pooling(grid_embeddings: np.ndarray) -> np.ndarray:
    """Apply column-wise mean pooling"""
    return grid_embeddings.mean(axis=0)


def hierarchical_token_pooling(
    arr: np.ndarray,
    pool_factor: int = 2,
    pooler: HierarchicalTokenPooler = None
) -> np.ndarray:
    """
    Apply hierarchical clustering to a single document embedding.
    """
    if pooler is None:
        pooler = HierarchicalTokenPooler()
    
    # Convert the array to 3D torch tensor
    arr_tensor = torch.from_numpy(arr[np.newaxis, :, :])
    # Apply hierarchical pooling
    pooled = pooler.pool_embeddings(arr_tensor, pool_factor=pool_factor)
    return pooled.cpu().detach().numpy()[0]


def demonstrate_pooling_methods(images_df: Any, logger: logging.Logger) -> None:
    """Demonstrate different pooling methods"""
    logger.info("=" * 80)
    logger.info("Demonstrating Pooling Methods")
    logger.info("=" * 80)
    
    try:
        example_embedding = images_df["image_embedding"].iloc[0]
        logger.info(f"Original embedding shape: {example_embedding.shape}")
        
        # Create image mask
        # Assuming the embedding is already masked, but let's demonstrate grid creation
        grid = embeddings_grid(example_embedding)
        logger.info(f"Grid shape: {grid.shape}")
        
        # Row pooling
        row_pooled = row_mean_pooling(grid)
        logger.info(f"Row-pooled shape: {row_pooled.shape}")
        
        # Column pooling
        col_pooled = column_mean_pooling(grid)
        logger.info(f"Column-pooled shape: {col_pooled.shape}")
        
        # Hierarchical pooling 2x
        pooler = HierarchicalTokenPooler()
        pooled_2x = hierarchical_token_pooling(example_embedding, pool_factor=2, pooler=pooler)
        logger.info(f"Hierarchical 2x pooled shape: {pooled_2x.shape}")
        
        # Hierarchical pooling 4x
        pooled_4x = hierarchical_token_pooling(example_embedding, pool_factor=4, pooler=pooler)
        logger.info(f"Hierarchical 4x pooled shape: {pooled_4x.shape}")
        
    except Exception as e:
        logger.warning(f"Could not demonstrate pooling methods: {e}", exc_info=True)


def create_optimized_collection(
    client: QdrantClient,
    logger: logging.Logger
) -> None:
    """Create Qdrant collection with multiple optimization variants"""
    logger.info("=" * 80)
    logger.info("Creating Optimized Collection")
    logger.info("=" * 80)
    
    try:
        # Delete existing collection
        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)
            logger.info(f"Deleted existing collection: {COLLECTION_NAME}")
        
        # Define vector configurations
        vectors_config = {
            "original": models.VectorParams(
                size=MODEL_DIM,
                distance=models.Distance.DOT,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
                hnsw_config=models.HnswConfigDiff(m=0),
                on_disk=True,
            ),
            "scalar_quantized": models.VectorParams(
                size=MODEL_DIM,
                distance=models.Distance.DOT,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
                hnsw_config=models.HnswConfigDiff(m=0),
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                    ),
                ),
                on_disk=True,
            ),
            "binary_quantized": models.VectorParams(
                size=MODEL_DIM,
                distance=models.Distance.DOT,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
                hnsw_config=models.HnswConfigDiff(m=0),
                quantization_config=models.BinaryQuantization(
                    binary=models.BinaryQuantizationConfig(
                        always_ram=True,
                    ),
                ),
                on_disk=True,
            ),
            "hierarchical_2x": models.VectorParams(
                size=MODEL_DIM,
                distance=models.Distance.DOT,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
                hnsw_config=models.HnswConfigDiff(m=0),
                on_disk=True,
            ),
            "hierarchical_4x": models.VectorParams(
                size=MODEL_DIM,
                distance=models.Distance.DOT,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
                hnsw_config=models.HnswConfigDiff(m=0),
                on_disk=True,
            ),
            "row_pooled": models.VectorParams(
                size=MODEL_DIM,
                distance=models.Distance.DOT,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
                hnsw_config=models.HnswConfigDiff(m=0),
                on_disk=True,
            ),
            "column_pooled": models.VectorParams(
                size=MODEL_DIM,
                distance=models.Distance.DOT,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
                hnsw_config=models.HnswConfigDiff(m=0),
                on_disk=True,
            ),
        }
        
        # Create collection
        client.create_collection(
            COLLECTION_NAME,
            vectors_config=vectors_config,
        )
        
        logger.info(f"Created collection: {COLLECTION_NAME}")
        logger.info(f"Vector variants: {len(vectors_config)}")
        for name in vectors_config.keys():
            logger.info(f"  - {name}")
        
    except Exception as e:
        logger.error(f"Failed to create collection: {e}", exc_info=True)
        raise


def populate_optimized_collection(
    client: QdrantClient,
    logger: logging.Logger
) -> None:
    """Populate collection with optimized embeddings"""
    logger.info("=" * 80)
    logger.info("Populating Optimized Collection")
    logger.info("=" * 80)
    
    try:
        logger.info(f"Load precomputed: {LOAD_PRECOMPUTED}")
        logger.info(f"Allowed documents: {ALLOWED_DOCS}")
        
        # Count total embeddings
        count = 0
        
        # Use the generator to load optimized embeddings efficiently
        for i, (image_path, vectors) in enumerate(
            tqdm(
                yield_optimized_embeddings(
                    load_precomputed=LOAD_PRECOMPUTED,
                    allowed_docs=ALLOWED_DOCS
                ),
                desc="Upserting embeddings"
            )
        ):
            # Insert each embedding with all optimization variants
            client.upsert(
                COLLECTION_NAME,
                points=[
                    models.PointStruct(
                        id=i,
                        vector={
                            "original": vectors["original"],
                            # Scalar and Binary Quantization are handled internally
                            # by Qdrant engine, so we send original vector
                            "scalar_quantized": vectors["original"],
                            "binary_quantized": vectors["original"],
                            "hierarchical_2x": vectors["hierarchical_2x"],
                            "hierarchical_4x": vectors["hierarchical_4x"],
                            "row_pooled": vectors["row_pooled"],
                            "column_pooled": vectors["column_pooled"],
                        },
                        payload={
                            "image_path": image_path,
                        },
                    )
                ],
            )
            count += 1
        
        logger.info(f"Successfully uploaded {count} documents with optimizations")
        
    except Exception as e:
        logger.error(f"Failed to populate collection: {e}", exc_info=True)
        raise


def load_queries(logger: logging.Logger) -> Tuple[List[str], List[np.ndarray]]:
    """Load query embeddings"""
    logger.info("=" * 80)
    logger.info("Loading Query Embeddings")
    logger.info("=" * 80)
    
    try:
        logger.info(f"Load precomputed: {LOAD_PRECOMPUTED}")
        queries_df = load_or_compute_query_embeddings(
            load_precomputed=LOAD_PRECOMPUTED,
        )
        
        logger.info(f"Loaded {len(queries_df)} queries")
        
        # Extract queries and embeddings
        queries = queries_df["query"].tolist()
        query_embeddings = queries_df["query_embedding"].tolist()
        
        # Log queries
        for i, query in enumerate(queries):
            logger.info(f"  Query {i + 1}: {query}")
        
        return queries, query_embeddings
        
    except Exception as e:
        logger.error(f"Failed to load queries: {e}", exc_info=True)
        raise


def run_optimization_comparison(
    queries: List[str],
    query_embeddings: List[np.ndarray],
    client: QdrantClient,
    logger: logging.Logger
) -> None:
    """Run comparison of optimization methods across queries"""
    logger.info("=" * 80)
    logger.info("Running Optimization Comparisons")
    logger.info("=" * 80)
    
    for idx, (query, query_embedding) in enumerate(zip(queries, query_embeddings)):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Query {idx + 1}: '{query}'")
        logger.info('=' * 80)
        
        try:
            # Compare optimization methods
            fig = compare_optimization_methods(
                query=query,
                query_embedding=query_embedding,
                client=client,
                collection_name=COLLECTION_NAME,
                vector_names=VECTOR_NAMES,
                limit=3,
            )
            
            # Save figure
            safe_query = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in query)
            safe_query = safe_query.replace(' ', '_')[:50]
            output_path = OUTPUT_DIR / f"optimization_comparison_query_{idx + 1}_{safe_query}.png"
            
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Saved comparison to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to compare optimizations for query {idx + 1}: {e}", exc_info=True)


def main():
    """Main execution function"""
    logger = setup_logging()
    
    logger.info("=" * 80)
    logger.info("ColPali Optimization Comparison System")
    logger.info("=" * 80)
    
    try:
        # Load processor
        processor = load_processor(logger)
        
        # Load image embeddings
        images_df = load_image_embeddings(logger)
        
        # Visualize sample images
        visualize_sample_images(images_df, logger)
        
        # Demonstrate concepts
        demonstrate_bucketing(logger)
        demonstrate_embedding_structure(images_df, processor, logger)
        demonstrate_pooling_methods(images_df, logger)
        
        # Connect to Qdrant
        logger.info(f"\nConnecting to Qdrant at: {QDRANT_URL}")
        client = QdrantClient(QDRANT_URL, timeout=120)
        logger.info("Successfully connected to Qdrant")
        
        # Create optimized collection
        create_optimized_collection(client, logger)
        
        # Populate collection
        populate_optimized_collection(client, logger)
        
        # Load queries
        queries, query_embeddings = load_queries(logger)
        
        # Run optimization comparisons
        run_optimization_comparison(queries, query_embeddings, client, logger)
        
        logger.info("=" * 80)
        logger.info("Execution completed successfully!")
        logger.info(f"All outputs saved to: {OUTPUT_DIR}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Fatal error during execution: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()