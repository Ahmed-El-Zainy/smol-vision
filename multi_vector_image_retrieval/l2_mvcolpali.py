"""
ColPali Image Retrieval System
Multi-vector image embeddings for document retrieval using vision-language models
"""

import logging
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from qdrant_client import QdrantClient, models

# Conditional imports based on CUDA availability
is_cuda_available = torch.cuda.is_available()

if is_cuda_available:
    from colpali_engine.models import ColPaliProcessor, ColPali
else:
    from colpali_engine.models import ColIdefics3Processor, ColIdefics3

from colpali_engine.interpretability import (
    get_similarity_maps_from_embeddings,
    plot_similarity_map,
)
from helper import (
    visualize_image_patches,
    load_or_compute_attention_embeddings,
    display_search_results,
)

# Configuration
SAMPLE_IMAGE_PATH = "ro_shared_data/attention-is-all-you-need/page-2.png"
COLLECTION_NAME = "colpali-experiments"
VECTOR_NAME = "colpali"
QDRANT_URL = "http://localhost:6333"
OUTPUT_DIR = Path("/mnt/user-data/outputs/colpali")

# Model configuration
MODEL_NAME_CUDA = "vidore/colpali-v1.3"
MODEL_NAME_CPU = "vidore/colSmol-256M"


def setup_logging() -> logging.Logger:
    """Configure logging with both file and console handlers"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(OUTPUT_DIR / "colpali_retrieval.log")
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


def load_model_and_processor(logger: logging.Logger) -> Tuple:
    """Load appropriate model and processor based on hardware availability"""
    logger.info("=" * 80)
    logger.info("Loading Model and Processor")
    logger.info("=" * 80)
    
    device = "cuda" if is_cuda_available else "cpu"
    logger.info(f"CUDA available: {is_cuda_available}")
    logger.info(f"Using device: {device}")
    
    try:
        if is_cuda_available:
            model_name = MODEL_NAME_CUDA
            logger.info(f"Loading ColPali model: {model_name}")
            processor = ColPaliProcessor.from_pretrained(model_name)
            model = ColPali.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
            )
        else:
            model_name = MODEL_NAME_CPU
            logger.info(f"Loading ColIdefics3 model: {model_name}")
            processor = ColIdefics3Processor.from_pretrained(model_name)
            model = ColIdefics3.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="eager",
            )
        
        logger.info(f"Model loaded successfully")
        logger.info(f"Model dimension: {model.dim}")
        if hasattr(model, 'patch_size'):
            logger.info(f"Patch size: {model.patch_size}")
        
        return model, processor, model_name
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def visualize_image_processing(
    image: Image.Image,
    processor,
    model,
    logger: logging.Logger
) -> None:
    """Visualize how the image is divided into patches"""
    logger.info("=" * 80)
    logger.info("Visualizing Image Patches")
    logger.info("=" * 80)
    
    logger.info(f"Original image size: {image.size}")
    
    try:
        patch_size = getattr(model, "patch_size", 0)
        fig = visualize_image_patches(
            image,
            processor,
            patch_size=patch_size,
            line_color="blue",
        )
        
        output_path = OUTPUT_DIR / "image_patches_visualization.png"
        fig.write_image(str(output_path))
        logger.info(f"Saved patch visualization to: {output_path}")
        
    except Exception as e:
        logger.warning(f"Could not save patch visualization: {e}")


def process_and_analyze_image(
    image: Image.Image,
    processor,
    model,
    logger: logging.Logger
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process image and extract embeddings"""
    logger.info("=" * 80)
    logger.info("Processing Image")
    logger.info("=" * 80)
    
    # Process image
    batch_images = processor.process_images([image]).to(model.device)
    logger.info(f"Batch keys: {list(batch_images.data.keys())}")
    logger.info(f"Number of tokens: {len(batch_images.input_ids[0])}")
    
    # Decode tokens for inspection
    decoded_tokens = processor.decode(batch_images.input_ids[0])
    logger.info(f"First 50 chars: {decoded_tokens[:50]}...")
    logger.info(f"Last 50 chars: ...{decoded_tokens[-50:]}")
    
    # Generate embeddings
    logger.info("Generating image embeddings...")
    with torch.no_grad():
        image_embeddings = model(**batch_images)
    
    logger.info(f"Image embeddings shape: {image_embeddings.shape}")
    
    # Apply image mask
    image_mask = processor.get_image_mask(batch_images)
    masked_image_embeddings = image_embeddings[image_mask]
    logger.info(f"Masked image embeddings shape: {masked_image_embeddings.shape}")
    
    return image_embeddings, image_mask


def process_query(
    query: str,
    processor,
    model,
    logger: logging.Logger
) -> Tuple[torch.Tensor, List[str]]:
    """Process query and return embeddings and tokens"""
    logger.info("=" * 80)
    logger.info(f"Processing Query: '{query}'")
    logger.info("=" * 80)
    
    # Process query
    batch_queries = processor.process_queries([query]).to(model.device)
    
    # Generate embeddings
    with torch.no_grad():
        query_embeddings = model(**batch_queries)
    
    logger.info(f"Query embeddings shape: {query_embeddings.shape}")
    
    # Clean and tokenize query
    query_content = processor.decode(batch_queries.input_ids[0])
    query_content = query_content.replace(processor.tokenizer.pad_token, "")
    query_content = query_content.replace(
        processor.query_augmentation_token, ""
    ).strip()
    
    query_tokens = processor.tokenizer.tokenize(query_content)
    logger.info(f"Query tokens ({len(query_tokens)}): {query_tokens}")
    
    # Log token indices
    for idx, token in enumerate(query_tokens):
        logger.info(f"  Token {idx}: {token}")
    
    return query_embeddings, query_tokens


def visualize_token_similarity(
    image: Image.Image,
    image_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    query_tokens: List[str],
    processor,
    model,
    image_mask: torch.Tensor,
    target_token: str,
    logger: logging.Logger
) -> None:
    """Visualize similarity map for a specific query token"""
    logger.info("=" * 80)
    logger.info(f"Visualizing Similarity for Token: '{target_token}'")
    logger.info("=" * 80)
    
    try:
        # Calculate number of patches
        n_patches = processor.get_n_patches(
            image_size=image.size,
            patch_size=getattr(model, "patch_size", 0),
        )
        logger.info(f"Number of patches: {n_patches}")
        
        # Get similarity maps
        similarity_maps = get_similarity_maps_from_embeddings(
            image_embeddings=image_embeddings,
            query_embeddings=query_embeddings,
            n_patches=n_patches,
            image_mask=image_mask,
        )
        
        # Find target token index
        target_token_idx = next(
            (idx for idx, val in enumerate(query_tokens) if target_token in val),
            None
        )
        
        if target_token_idx is None:
            logger.warning(f"Token '{target_token}' not found in query tokens")
            return
        
        logger.info(f"Target token index: {target_token_idx}")
        
        # Get similarity map for target token
        similarity_mask = similarity_maps[0][target_token_idx]
        
        # Plot and save
        fig = plot_similarity_map(
            image=image,
            similarity_map=similarity_mask,
            figsize=(8, 8),
            show_colorbar=False,
        )
        
        output_path = OUTPUT_DIR / f"similarity_map_{target_token}.png"
        fig.savefig(output_path, bbox_inches='tight', dpi=150)
        logger.info(f"Saved similarity map to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to visualize similarity: {e}", exc_info=True)


def create_qdrant_collection(
    client: QdrantClient,
    model,
    logger: logging.Logger
) -> None:
    """Create Qdrant collection for ColPali embeddings"""
    logger.info("=" * 80)
    logger.info("Creating Qdrant Collection")
    logger.info("=" * 80)
    
    try:
        # Delete existing collection if present
        if client.collection_exists(COLLECTION_NAME):
            client.delete_collection(COLLECTION_NAME)
            logger.info(f"Deleted existing collection: {COLLECTION_NAME}")
        
        # Create collection
        client.create_collection(
            COLLECTION_NAME,
            vectors_config={
                VECTOR_NAME: models.VectorParams(
                    size=model.dim,
                    distance=models.Distance.DOT,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM,
                    ),
                    hnsw_config=models.HnswConfigDiff(m=0),
                )
            },
        )
        
        logger.info(f"Created collection: {COLLECTION_NAME}")
        logger.info(f"  - Vector name: {VECTOR_NAME}")
        logger.info(f"  - Vector size: {model.dim}")
        logger.info(f"  - Distance: DOT")
        logger.info(f"  - Comparator: MAX_SIM")
        
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        raise


def populate_collection(
    client: QdrantClient,
    model_name: str,
    logger: logging.Logger,
    load_precomputed: bool = True
) -> None:
    """Load embeddings and populate Qdrant collection"""
    logger.info("=" * 80)
    logger.info("Populating Collection with Document Embeddings")
    logger.info("=" * 80)
    
    try:
        # Load or compute embeddings
        logger.info(f"Loading embeddings (precomputed={load_precomputed})...")
        embeddings_df = load_or_compute_attention_embeddings(
            load_precomputed=load_precomputed,
            model_name=model_name,
        )
        
        logger.info(f"Loaded {len(embeddings_df)} document embeddings")
        
        # Upsert to Qdrant
        logger.info("Upserting embeddings to Qdrant...")
        for idx, row in tqdm(embeddings_df.iterrows(), total=len(embeddings_df), desc="Uploading"):
            client.upsert(
                COLLECTION_NAME,
                points=[
                    models.PointStruct(
                        id=uuid.uuid4().hex,
                        vector={
                            VECTOR_NAME: row["image_embedding"],
                        },
                        payload={
                            "file_path": row["file_path"],
                        },
                    )
                ],
            )
        
        logger.info(f"Successfully uploaded {len(embeddings_df)} documents")
        
    except Exception as e:
        logger.error(f"Failed to populate collection: {e}", exc_info=True)
        raise


def search_documents(
    query: str,
    processor,
    model,
    client: QdrantClient,
    limit: int = 3,
    logger: Optional[logging.Logger] = None
) -> List[models.ScoredPoint]:
    """Search for documents using ColPali embeddings"""
    if logger:
        logger.info(f"Searching for: '{query}' (limit={limit})")
    
    try:
        # Process query
        batch_queries = processor.process_queries([query]).to(model.device)
        with torch.no_grad():
            query_embeddings = model(**batch_queries).to(dtype=torch.float32)
        
        # Search
        results = client.query_points(
            COLLECTION_NAME,
            query=query_embeddings[0].cpu().numpy(),
            using=VECTOR_NAME,
            limit=limit,
            with_payload=True,
        ).points
        
        if logger:
            logger.info(f"Found {len(results)} results")
            for i, result in enumerate(results, 1):
                logger.info(f"  {i}. Score: {result.score:.4f}, Path: {result.payload.get('file_path', 'N/A')}")
        
        return results
        
    except Exception as e:
        if logger:
            logger.error(f"Search failed: {e}", exc_info=True)
        raise


def visualize_search_results(
    results: List[models.ScoredPoint],
    query: str,
    layout: str,
    logger: logging.Logger
) -> None:
    """Visualize and save search results"""
    logger.info(f"Visualizing search results for: '{query}'")
    
    try:
        fig = display_search_results(results, layout=layout)
        
        # Save figure
        safe_query = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in query)
        safe_query = safe_query.replace(' ', '_')[:50]
        output_path = OUTPUT_DIR / f"search_results_{safe_query}.png"
        
        fig.savefig(output_path, bbox_inches='tight', dpi=150)
        logger.info(f"Saved search results to: {output_path}")
        
    except Exception as e:
        logger.warning(f"Could not save search results visualization: {e}")


def run_demonstration(
    image_path: str,
    processor,
    model,
    logger: logging.Logger
) -> None:
    """Run demonstration of image processing and query analysis"""
    logger.info("=" * 80)
    logger.info("DEMONSTRATION: Image Processing and Query Analysis")
    logger.info("=" * 80)
    
    # Load image
    try:
        image = Image.open(image_path)
        logger.info(f"Loaded image from: {image_path}")
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        return
    
    # Visualize patches
    visualize_image_processing(image, processor, model, logger)
    
    # Process image
    image_embeddings, image_mask = process_and_analyze_image(
        image, processor, model, logger
    )
    
    # Process query
    query = "How does a single transformer layer look like?"
    query_embeddings, query_tokens = process_query(
        query, processor, model, logger
    )
    
    # Visualize similarity for specific token
    visualize_token_similarity(
        image=image,
        image_embeddings=image_embeddings,
        query_embeddings=query_embeddings,
        query_tokens=query_tokens,
        processor=processor,
        model=model,
        image_mask=image_mask,
        target_token="layer",
        logger=logger
    )


def run_search_experiments(
    processor,
    model,
    client: QdrantClient,
    logger: logging.Logger
) -> None:
    """Run search experiments with different queries"""
    logger.info("=" * 80)
    logger.info("SEARCH EXPERIMENTS")
    logger.info("=" * 80)
    
    queries = [
        "model architecture",
        "scaled dot-product attention",
    ]
    
    for query in queries:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Query: '{query}'")
        logger.info('=' * 80)
        
        results = search_documents(
            query=query,
            processor=processor,
            model=model,
            client=client,
            limit=3,
            logger=logger
        )
        
        visualize_search_results(
            results=results,
            query=query,
            layout="horizontal",
            logger=logger
        )


def main():
    """Main execution function"""
    logger = setup_logging()
    
    logger.info("=" * 80)
    logger.info("ColPali Image Retrieval System")
    logger.info("=" * 80)
    
    try:
        # Load model and processor
        model, processor, model_name = load_model_and_processor(logger)
        
        # Run demonstration
        run_demonstration(
            image_path=SAMPLE_IMAGE_PATH,
            processor=processor,
            model=model,
            logger=logger
        )
        
        # Connect to Qdrant
        logger.info(f"\nConnecting to Qdrant at: {QDRANT_URL}")
        client = QdrantClient(QDRANT_URL)
        logger.info("Successfully connected to Qdrant")
        
        # Create collection
        create_qdrant_collection(client, model, logger)
        
        # Populate collection
        populate_collection(
            client=client,
            model_name=model_name,
            logger=logger,
            load_precomputed=True  # Set to False to regenerate embeddings
        )
        
        # Run search experiments
        run_search_experiments(processor, model, client, logger)
        
        logger.info("=" * 80)
        logger.info("Execution completed successfully!")
        logger.info(f"All outputs saved to: {OUTPUT_DIR}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Fatal error during execution: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()