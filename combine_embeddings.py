import os
import pickle
import numpy as np
import faiss
from collections import defaultdict

# Paths
PROGRESS_DIR = "progress"
EMBEDDINGS_PROGRESS_PATH = os.path.join(PROGRESS_DIR, "embeddings_progress.pkl")
EXISTING_INDEX_PATH = "clip_product.index"
EXISTING_IDS_PATH = "image_ids.npy"
NEW_INDEX_PATH = "clip_product_combined.index"
NEW_IDS_PATH = "image_ids_combined.npy"

def load_existing_index():
    """Load existing FAISS index and image IDs"""
    print("Loading existing FAISS index...")
    if os.path.exists(EXISTING_INDEX_PATH):
        index = faiss.read_index(EXISTING_INDEX_PATH)
        image_ids = np.load(EXISTING_IDS_PATH)
        print(f"Loaded existing index with {len(image_ids)} images")
        return index, image_ids
    return None, None

def load_new_embeddings():
    """Load newly processed embeddings"""
    print("Loading new embeddings...")
    if os.path.exists(EMBEDDINGS_PROGRESS_PATH):
        with open(EMBEDDINGS_PROGRESS_PATH, 'rb') as f:
            progress_data = pickle.load(f)
        
        # Convert defaultdict to regular dict for easier handling
        embeddings_dict = dict(progress_data['product_embeddings'])
        print(f"Loaded {len(embeddings_dict)} new products")
        return embeddings_dict
    return None

def combine_embeddings():
    """Combine existing and new embeddings into a new FAISS index"""
    # Load existing index and IDs
    existing_index, existing_ids = load_existing_index()
    if existing_index is None:
        print("No existing index found. Creating new index...")
        dimension = 512  # CLIP embedding dimension
        index = faiss.IndexFlatL2(dimension)
        existing_ids = np.array([])
    
    # Load new embeddings
    new_embeddings = load_new_embeddings()
    if new_embeddings is None:
        print("No new embeddings found!")
        return
    
    # Prepare new embeddings
    all_embeddings = []
    all_ids = []
    
    # Add existing embeddings
    if len(existing_ids) > 0:
        all_embeddings.append(existing_index.reconstruct_n(0, existing_index.ntotal))
        all_ids.extend(existing_ids)
    
    # Add new embeddings
    for product_id, embeddings in new_embeddings.items():
        for emb in embeddings:
            all_embeddings.append(emb)
            all_ids.append(product_id)
    
    # Convert to numpy arrays
    all_embeddings = np.vstack(all_embeddings)
    all_ids = np.array(all_ids)
    
    # Create new index
    print("Creating new combined index...")
    new_index = faiss.IndexFlatL2(all_embeddings.shape[1])
    new_index.add(all_embeddings)
    
    # Save new index and IDs
    print("Saving new index and IDs...")
    faiss.write_index(new_index, NEW_INDEX_PATH)
    np.save(NEW_IDS_PATH, all_ids)
    
    print(f"\nCombined index created successfully!")
    print(f"Total images in new index: {len(all_ids)}")
    print(f"New index saved to: {NEW_INDEX_PATH}")
    print(f"New IDs saved to: {NEW_IDS_PATH}")

if __name__ == "__main__":
    combine_embeddings() 