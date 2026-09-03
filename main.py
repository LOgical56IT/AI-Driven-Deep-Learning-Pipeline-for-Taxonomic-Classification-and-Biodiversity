from data_loader import load_fasta
from kmer_encoder import encode_sequences
from clustering import cluster_sequences
from config import KMER_SIZE

# Load sequences
sequences = load_fasta("data/reads.fasta")
print("Loaded sequences:", len(sequences))

# Encode sequences
X = encode_sequences(sequences, KMER_SIZE)
print("Encoded feature shape:", X.shape)

# Cluster
labels, embedding = cluster_sequences(X)

num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
noise_count = (labels == -1).sum()

print("Clusters discovered:", num_clusters)
print("Noise sequences:", noise_count)
