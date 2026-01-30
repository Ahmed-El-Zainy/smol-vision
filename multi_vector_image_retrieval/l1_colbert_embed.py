"""
ColBERT and Dense Embedding Retrieval with Qdrant
Demonstrates multi-vector and dense vector search with visualization
"""


import logging
import time
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from fastembed import LateInteractionTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models
from helper import (
    tokenize_late_interaction,
    visualize_maxsim_matrix,
    display_results_side_by_side
)


# Configuration
COLLECTION_NAME = "colbert-tests"
DENSE_VECTOR_NAME = "BAAI-bge-small-en-v1.5"
COLBERT_VECTOR_NAME = "colbert-ir-colbertv2.0"
MODEL_NAME_COLBERT = "colbert-ir/colbertv2.0"
MODEL_NAME_DENSE = "BAAI/bge-small-en-v1.5"
QDRANT_URL = "http://localhost:6333"
OUTPUT_DIR = Path("multi_vector_image_retrieval/outputs")


def setup_logging() -> logging.Logger:
    """Configure logging with both file and console handlers"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(OUTPUT_DIR / "colbert_retrieval.log")
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


def init_colbert_model(model_name: str, logger: logging.Logger) -> LateInteractionTextEmbedding:
    """Initialize ColBERT model"""
    logger.info(f"Initializing ColBERT model: {model_name}")
    try:
        model = LateInteractionTextEmbedding(model_name)
        logger.info(f"ColBERT model loaded successfully. Embedding size: {model.embedding_size}")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize ColBERT model: {e}")
        raise


def init_dense_model(model_name: str, logger: logging.Logger) -> TextEmbedding:
    """Initialize dense embedding model"""
    logger.info(f"Initializing dense model: {model_name}")
    try:
        model = TextEmbedding(model_name)
        logger.info(f"Dense model loaded successfully. Embedding size: {model.embedding_size}")
        return model
    except Exception as e:
        logger.error(f"Failed to initialize dense model: {e}")
        raise


def demonstrate_colbert_scoring(
    colbert_model: LateInteractionTextEmbedding,
    logger: logging.Logger
) -> None:
    """Demonstrate ColBERT scoring with a sample document"""
    logger.info("=" * 80)
    logger.info("DEMONSTRATION: ColBERT Scoring")
    logger.info("=" * 80)
    
    document = """This study examines the environmental benefits of 
electric bus fleets in three major metropolitan areas over a 
two-year period. Our analysis shows that electric buses reduce 
carbon emissions by an average of 65% compared to traditional 
diesel buses, while also decreasing noise pollution in urban 
centers by 40 decibels."""
    
    # Tokenize document
    document_tokens = tokenize_late_interaction(colbert_model, document)
    logger.info(f"Document tokens ({len(document_tokens)} tokens): {document_tokens}")
    
    # Get document embeddings
    document_embeddings = next(colbert_model.passage_embed([document]))
    logger.info(f"Document embeddings shape: {document_embeddings.shape}")
    
    # Query
    query = "advantages of EV cars"
    query_tokens = tokenize_late_interaction(colbert_model, query, is_doc=False)
    logger.info(f"Query: '{query}'")
    logger.info(f"Query tokens ({len(query_tokens)} tokens): {query_tokens}")
    
    # Get query embeddings
    query_embeddings = next(colbert_model.query_embed([query]))
    logger.info(f"Query embeddings shape: {query_embeddings.shape}")
    
    # Calculate similarity matrix
    similarity_matrix = np.dot(query_embeddings, document_embeddings.T)
    logger.info(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    # Calculate MaxSim score
    score = similarity_matrix.max(axis=1).sum()
    logger.info(f"MaxSim score: {score:.4f}")
    
    # Visualize and save
    try:
        fig = visualize_maxsim_matrix(
            similarity_matrix,
            query_tokens,
            document_tokens=document_tokens,
            width=600
        )
        output_path = OUTPUT_DIR / "demo_similarity_matrix.png"
        fig.write_image(str(output_path))
        logger.info(f"Saved similarity matrix visualization to: {output_path}")
    except Exception as e:
        logger.warning(f"Could not save similarity matrix image: {e}")


def create_qdrant_collection(
    client: QdrantClient,
    colbert_model: LateInteractionTextEmbedding,
    dense_model: TextEmbedding,
    logger: logging.Logger
) -> None:
    """Create Qdrant collection with multi-vector and dense vector configurations"""
    logger.info("=" * 80)
    logger.info("Creating Qdrant Collection")
    logger.info("=" * 80)
    
    try:
        # Delete existing collection if present
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info(f"Deleted existing collection: {COLLECTION_NAME}")
        except Exception:
            logger.info(f"Collection {COLLECTION_NAME} does not exist, creating new")
        
        # Create collection
        client.create_collection(
            COLLECTION_NAME,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=dense_model.embedding_size,
                    distance=models.Distance.COSINE,
                ),
                COLBERT_VECTOR_NAME: models.VectorParams(
                    size=colbert_model.embedding_size,
                    distance=models.Distance.DOT,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM,
                    ),
                    hnsw_config=models.HnswConfigDiff(m=0),
                )
            }
        )
        logger.info(f"Created collection: {COLLECTION_NAME}")
        logger.info(f"  - Dense vector: {DENSE_VECTOR_NAME} (size: {dense_model.embedding_size})")
        logger.info(f"  - ColBERT vector: {COLBERT_VECTOR_NAME} (size: {colbert_model.embedding_size})")
        
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        raise


def add_documents_to_collection(
    client: QdrantClient,
    colbert_model: LateInteractionTextEmbedding,
    dense_model: TextEmbedding,
    logger: logging.Logger
) -> None:
    """Add documents to Qdrant collection"""
    logger.info("=" * 80)
    logger.info("Adding Documents to Collection")
    logger.info("=" * 80)
    
    documents = [
        "Qdrant is a vector database designed for similarity search applications",
        "SQL databases use structured tables with predefined schemas for data storage",
        "Using Qdrant you can store embeddings and perform efficient searches",
        "Traditional SQL queries filter data using exact matches and joins",
        "Qdrant supports multi-vector configurations for late interaction models like ColBERT",
        "SQL performs well for transactional workloads but lacks semantic search capabilities",
        "The Qdrant client allows you to create collections with custom distance metrics",
        "Migrating from SQL to vector databases enables similarity-based retrieval at scale",
        "Qdrant's MaxSim comparator enables token-level similarity scoring for multi-vectors",
        "SQL databases struggle with high-dimensional embeddings unlike specialized vector stores",
    ]
    
    try:
        points = []
        for i, document in enumerate(documents, start=1):
            point = models.PointStruct(
                id=i,
                vector={
                    DENSE_VECTOR_NAME: next(dense_model.passage_embed([document])).tolist(),
                    COLBERT_VECTOR_NAME: next(colbert_model.passage_embed([document])).tolist(),
                },
                payload={"text": document},
            )
            points.append(point)
        
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        logger.info(f"Successfully added {len(documents)} documents to collection")
        
    except Exception as e:
        logger.error(f"Failed to add documents: {e}")
        raise


def colbert_query(
    client: QdrantClient,
    colbert_model: LateInteractionTextEmbedding,
    query: str,
    limit: int = 5,
    logger: Optional[logging.Logger] = None
) -> List[Dict]:
    """Perform ColBERT query"""
    if logger:
        logger.info(f"Executing ColBERT query: '{query}'")
    
    start_time = time.monotonic()
    embedding = next(colbert_model.query_embed([query]))
    embed_time = time.monotonic() - start_time
    
    if logger:
        logger.info(f"  ColBERT embedding generation time: {embed_time:.4f}s")
    
    start_time = time.monotonic()
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding.tolist(),
        using=COLBERT_VECTOR_NAME,
        limit=limit,
        with_payload=True,
    )
    query_time = time.monotonic() - start_time
    
    if logger:
        logger.info(f"  Query execution time: {query_time:.4f}s")
        logger.info(f"  Retrieved {len(result.points)} results")
    
    return [point.payload for point in result.points]


def dense_query(
    client: QdrantClient,
    dense_model: TextEmbedding,
    query: str,
    limit: int = 5,
    logger: Optional[logging.Logger] = None
) -> List[Dict]:
    """Perform dense embedding query"""
    if logger:
        logger.info(f"Executing Dense query: '{query}'")
    
    start_time = time.monotonic()
    embedding = next(dense_model.query_embed([query]))
    embed_time = time.monotonic() - start_time
    
    if logger:
        logger.info(f"  Dense embedding generation time: {embed_time:.4f}s")
    
    start_time = time.monotonic()
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding.tolist(),
        using=DENSE_VECTOR_NAME,
        limit=limit,
        with_payload=True,
    )
    query_time = time.monotonic() - start_time
    
    if logger:
        logger.info(f"  Query execution time: {query_time:.4f}s")
        logger.info(f"  Retrieved {len(result.points)} results")
    
    return [point.payload for point in result.points]


def compare_retrieval_methods(
    client: QdrantClient,
    colbert_model: LateInteractionTextEmbedding,
    dense_model: TextEmbedding,
    query: str,
    logger: logging.Logger
) -> None:
    """Compare ColBERT and Dense retrieval methods"""
    logger.info("=" * 80)
    logger.info("Comparing Retrieval Methods")
    logger.info("=" * 80)
    
    # Tokenize query
    query_tokens = tokenize_late_interaction(colbert_model, query, is_doc=False)
    logger.info(f"Query: '{query}'")
    logger.info(f"Query tokens: {query_tokens}")
    
    # Execute queries
    colbert_hits = colbert_query(client, colbert_model, query, logger=logger)
    dense_hits = dense_query(client, dense_model, query, logger=logger)
    
    # Log results
    logger.info("\nColBERT Results:")
    for i, hit in enumerate(colbert_hits, 1):
        logger.info(f"  {i}. {hit.get('text', 'N/A')}")
    
    logger.info("\nDense Results:")
    for i, hit in enumerate(dense_hits, 1):
        logger.info(f"  {i}. {hit.get('text', 'N/A')}")
    
    # Visualize side by side
    try:
        fig = display_results_side_by_side(
            left_results=colbert_hits,
            right_results=dense_hits,
            left_title="ColBERT Results",
            right_title="Dense Results",
            query=query,
        )
        output_path = OUTPUT_DIR / "retrieval_comparison.png"
        fig.write_image(str(output_path))
        logger.info(f"\nSaved retrieval comparison to: {output_path}")
    except Exception as e:
        logger.warning(f"Could not save comparison image: {e}")
    
    return colbert_hits, query_tokens


def analyze_top_result(
    colbert_model: LateInteractionTextEmbedding,
    colbert_hits: List[Dict],
    query: str,
    query_tokens: List[str],
    logger: logging.Logger
) -> None:
    """Analyze the top ColBERT result with detailed scoring"""
    logger.info("=" * 80)
    logger.info("Analyzing Top ColBERT Result")
    logger.info("=" * 80)
    
    if not colbert_hits:
        logger.warning("No ColBERT results to analyze")
        return
    
    top_document = colbert_hits[0]["text"]
    logger.info(f"Top document: {top_document}")
    
    # Tokenize top document
    top_document_tokens = tokenize_late_interaction(colbert_model, top_document)
    logger.info(f"Document tokens ({len(top_document_tokens)} tokens): {top_document_tokens}")
    
    # Get embeddings
    top_document_vector = next(colbert_model.passage_embed([top_document]))
    query_vector = next(colbert_model.query_embed([query]))
    
    # Calculate similarity matrix
    similarity_matrix = np.dot(query_vector, top_document_vector.T)
    logger.info(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    # Calculate score
    score = similarity_matrix.max(axis=1).sum()
    logger.info(f"MaxSim score: {score:.4f}")
    
    # Visualize and save
    try:
        fig = visualize_maxsim_matrix(
            similarity_matrix,
            query_tokens=query_tokens,
            document_tokens=top_document_tokens,
            width=600,
        )
        output_path = OUTPUT_DIR / "top_result_similarity_matrix.png"
        fig.write_image(str(output_path))
        logger.info(f"Saved top result similarity matrix to: {output_path}")
    except Exception as e:
        logger.warning(f"Could not save top result image: {e}")


def main():
    """Main execution function"""
    logger = setup_logging()
    
    logger.info("=" * 80)
    logger.info("ColBERT and Dense Embedding Retrieval System")
    logger.info("=" * 80)
    
    try:
        # Initialize models
        colbert_model = init_colbert_model(MODEL_NAME_COLBERT, logger)
        dense_model = init_dense_model(MODEL_NAME_DENSE, logger)
        
        # Demonstrate ColBERT scoring
        demonstrate_colbert_scoring(colbert_model, logger)
        
        # Connect to Qdrant
        logger.info(f"\nConnecting to Qdrant at: {QDRANT_URL}")
        client = QdrantClient(QDRANT_URL)
        logger.info("Successfully connected to Qdrant")
        
        # Create collection
        create_qdrant_collection(client, colbert_model, dense_model, logger)
        
        # Add documents
        add_documents_to_collection(client, colbert_model, dense_model, logger)
        
        # Compare retrieval methods
        query = "search performance in Qdrant"
        colbert_hits, query_tokens = compare_retrieval_methods(
            client, colbert_model, dense_model, query, logger
        )
        
        # Analyze top result
        analyze_top_result(colbert_model, colbert_hits, query, query_tokens, logger)
        
        logger.info("=" * 80)
        logger.info("Execution completed successfully!")
        logger.info(f"All outputs saved to: {OUTPUT_DIR}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Fatal error during execution: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()