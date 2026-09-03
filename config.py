KMER_SIZE = 6
MIN_CLUSTER_SIZE = 50

# Optional: path to a local BLAST+ nucleotide database (e.g. SILVA/PR2/NCBI nt)
# Download from `https://ftp.ncbi.nlm.nih.gov/blast/db/` and build locally,
# then set BLAST_DB to the basename of your database (without file extensions).
# Leave empty string to disable BLAST-based taxonomic annotation.
BLAST_DB = ""  # e.g. r"C:/blast/db/nt/nt"# Maximum number of BLAST hits to parse per query
BLAST_MAX_HITS = 5

# Minimum percent identity to consider a hit informative
BLAST_MIN_IDENTITY = 80.0
