from colpali_engine.models import ColPaliProcessor
from helper import load_sample_image_embeddings
import matplotlib.pyplot as plt
from PIL import Image
from qdrant_client import QdrantClient, models
## performing hich pooling 
from colpali_engine.compression.token_pooling import (
    HierarchicalTokenPooler,)
import numpy as np
import torch
from helper import load_or_compute_query_embeddings


LOAD_PRECOMPUTED = True

model_name = "vidore/colpali-v1.3"
processor = ColPaliProcessor.from_pretrained(model_name)
# Load or compute image embeddings using helper function
# that only loads a sample of data
images_df = load_sample_image_embeddings(
    load_precomputed=LOAD_PRECOMPUTED,)


# Display some random entries
sample_df = images_df.sample(n=3, random_state=42)
sample_df



fig, axarr = plt.subplots(3, 1, figsize=(9, 16))
for i, (_, row) in enumerate(sample_df.iterrows()):
    image = Image.open(row["image_path"])
    axarr[i].imshow(image.convert("RGB"))
    
images_df["original_shape"] = images_df["image_embedding"].apply(
    lambda x: x.shape
)
images_df.sample(n=5)

import numpy as np

# Perform bucketing
min_value, max_value = -0.8, 1.2
value_space = np.linspace(min_value, max_value, 256)

# Find which bucket index the target value belongs to
target_value = 0.2
bucket_index = np.argmin(np.abs(value_space - target_value))
bucket_index

simple_vector = 2 * np.random.random(16) - 1.0
binary_vector = (simple_vector > 0).astype(int)
print(simple_vector, "->", binary_vector)



## display img token and embedding vectors
image = Image.open(images_df["image_path"][0])
batch_images = processor.process_images([image])
processor.decode(batch_images.input_ids[0])

image_mask = processor.get_image_mask(batch_images)[0]
images_df["image_embedding"][0][image_mask]

patch_size = 32  # the default Colpali patch size
model_dim = 128  # the default Colpali embedding size

def embeddings_grid(image_embeddings: np.ndarray):
    """
    Reshape the image embeddings so we have a grid of patches
    and their corresponding embeddings.
    """
    return image_embeddings.reshape((patch_size, patch_size, model_dim))


grid = embeddings_grid(images_df["image_embedding"][0][image_mask])
grid


## performing row & column
def row_mean_pooling(grid_embeddings: np.ndarray) -> np.ndarray:
    return grid_embeddings.mean(axis=1)

def column_mean_pooling(grid_embeddings: np.ndarray) -> np.ndarray:
    return grid_embeddings.mean(axis=0)
row_mean_pooling(grid)
column_mean_pooling(grid)



pooler = HierarchicalTokenPooler()

def hierarchical_token_pooling(
    arr: np.ndarray, pool_factor: int = 2
) -> np.ndarray:
    """
    Apply hierarchical clustering to a single document embedding.
    """
    # Convert the array to 3D torch tensor
    arr_tensor = torch.from_numpy(arr[np.newaxis, :, :])
    # Apply hierarchical pooling
    pooled = pooler.pool_embeddings(arr_tensor, pool_factor=pool_factor)
    return pooled.cpu().detach().numpy()[0]


# Demonstrate on a single example
example_embedding = images_df["image_embedding"][0]
print(f"Original shape: {example_embedding.shape}")
# Apply hierarchical token pooling with pool_factor=2
pooled_embedding = hierarchical_token_pooling(
    example_embedding, pool_factor=2)
print(f"Pooled shape: {pooled_embedding.shape}")


collection_name = "colpali-optimizations"

# Connect to Qdrant
client = QdrantClient("http://localhost:6333", timeout=120)

# Delete collection if it exists
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)


client.create_collection(
    collection_name,
    vectors_config={
        "original": models.VectorParams(
            size=128,
            distance=models.Distance.DOT,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
            hnsw_config=models.HnswConfigDiff(m=0),
            on_disk=True,
        ),
        
        "scalar_quantized": models.VectorParams(
            size=128,
            distance=models.Distance.DOT,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
            hnsw_config=models.HnswConfigDiff(m=0),
            # Enable Scalar Quantization for a single named vector
            # and not the entire collection
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                ),
            ),
            on_disk=True,
        ),
        "binary_quantized": models.VectorParams(
            size=128,
            distance=models.Distance.DOT,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
            hnsw_config=models.HnswConfigDiff(m=0),
            # Enable Binary Quantization for a single named vector
            # and not the entire collection
            quantization_config=models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(
                    always_ram=True,
                ),
            ),
            on_disk=True,
        ),
        "hierarchical_2x": models.VectorParams(
            size=128,
            distance=models.Distance.DOT,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
            hnsw_config=models.HnswConfigDiff(m=0),
            on_disk=True,
        ),
        "hierarchical_4x": models.VectorParams(
            size=128,
            distance=models.Distance.DOT,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
            hnsw_config=models.HnswConfigDiff(m=0),
            on_disk=True,
        ),
        "row_pooled": models.VectorParams(
            size=128,
            distance=models.Distance.DOT,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
            hnsw_config=models.HnswConfigDiff(m=0),
            on_disk=True,
        ),
        "column_pooled": models.VectorParams(
            size=128,
            distance=models.Distance.DOT,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM,
            ),
            hnsw_config=models.HnswConfigDiff(m=0),
            on_disk=True,
        ),
    },
)




from helper import yield_optimized_embeddings
from tqdm import tqdm

allowed_docs = (
    "AI4E_W1", "AI4E_W2", "DLS_C4W4", "GenAI4E_W1",
    "MLS_C2_W1", "MLS_C3_W2", "MLS_C2_W3", "DLS_C3_W1",
    "DLS_C1_W1", "MLS_C3_W3", "MLS_C1_W1",
)

# Use the generator to load optimized embeddings efficiently
# Set load_precomputed=False the first time to compute and save optimizations
# Set load_precomputed=True to load from saved files (much faster!)
for i, (image_path, vectors) in enumerate(
    tqdm(
        yield_optimized_embeddings(load_precomputed=LOAD_PRECOMPUTED, 
                                   allowed_docs=allowed_docs),
        desc="Upserting embeddings"
    )
):
    # Insert each embedding with all optimization variants
    client.upsert(
        collection_name,
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


## loading sample query vectors
## load or compute query embeddings using helper func
queries_df = load_or_compute_query_embeddings(
    load_precomputed=LOAD_PRECOMPUTED,
)

## extract queries and query embeddings for later use
queries = queries_df["query"].tolist()
query_embeddings = queries_df["query_embedding"].tolist()
## display the queries
queries_df
# Define all optimization methods to compare
vector_names = [
    "original",  # Always first (baseline)
    "scalar_quantized",
    "binary_quantized",
    "row_pooled",
    "column_pooled",
    "hierarchical_2x",
    "hierarchical_4x",
]


## compare diff optimziation methods
from helper import compare_optimization_methods

# Query 1: Coffee mug
fig = compare_optimization_methods(
    query=queries[0],
    query_embedding=query_embeddings[0],
    client=client,
    collection_name=collection_name,
    vector_names=vector_names,
    limit=3,
)
fig.show()


# Query 2: Size vs performance tradeoff
fig = compare_optimization_methods(
    query=queries[1],
    query_embedding=query_embeddings[1],
    client=client,
    collection_name=collection_name,
    vector_names=vector_names,
    limit=3,
)
fig.show()


# Query 3: One learning algorithm
fig = compare_optimization_methods(
    query=queries[2],
    query_embedding=query_embeddings[2],
    client=client,
    collection_name=collection_name,
    vector_names=vector_names,
    limit=3,
)
fig.show()


