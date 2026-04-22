import numpy as np
import pickle
import os


def build_adj_from_string(geneList, string_links_path=None, string_info_path=None,
                           threshold=700, cache_path=None):
    """
    Build adjacency matrix from STRING protein interaction database.

    geneList:          sorted list of gene symbols (from totGeneSet)
    string_links_path: path to 9606.protein.links.v12.0.txt
    string_info_path:  path to 9606.protein.info.v12.0.txt  (ENSP -> gene symbol)
    threshold:         combined score cutoff (0-1000). 700 = high confidence
    cache_path:        save/load from pickle to avoid recomputation
    """
    N = len(geneList)
    gene_set = set(geneList)
    gene_idx = {g: i for i, g in enumerate(geneList)}

    if cache_path and os.path.exists(cache_path):
        print(f"Loading adjacency matrix from cache: {cache_path}")
        return pickle.load(open(cache_path, 'rb'))

    adj = np.eye(N, dtype=np.float32)  # start with self-loops only

    # --- guard: both files must exist ---
    if string_links_path is None or not os.path.exists(string_links_path):
        print("WARNING: STRING links file not found. Using identity adjacency.")
        print(f"  Expected: {string_links_path}")
        return adj

    if string_info_path is None or not os.path.exists(string_info_path):
        print("WARNING: STRING info file not found. Using identity adjacency.")
        print(f"  Expected: {string_info_path}")
        return adj

    # --- Step 1: build ENSP -> gene symbol map from info file ---
    print("Reading protein info file (ENSP -> gene symbol)...")
    ensp_to_gene = {}
    with open(string_info_path) as f:
        next(f)  # skip header: #string_protein_id preferred_name ...
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            ensp = parts[0].replace("9606.", "")   # e.g. ENSP00000000233
            symbol = parts[1].strip()               # e.g. ARF5
            ensp_to_gene[ensp] = symbol

    in_our_list = sum(1 for s in ensp_to_gene.values() if s in gene_set)
    print(f"  {len(ensp_to_gene)} proteins in info file, "
          f"{in_our_list} map to our {N}-gene list")

    # --- Step 2: scan links file and fill adjacency ---
    print(f"Building adjacency from STRING links (threshold={threshold})...")
    matched = 0
    total_lines = 0
    with open(string_links_path) as f:
        next(f)  # skip header: protein1 protein2 combined_score
        for line in f:
            total_lines += 1
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            ensp_a = parts[0].replace("9606.", "")
            ensp_b = parts[1].replace("9606.", "")
            score  = int(parts[2])

            if score < threshold:
                continue

            gene_a = ensp_to_gene.get(ensp_a)
            gene_b = ensp_to_gene.get(ensp_b)

            if gene_a in gene_set and gene_b in gene_set:
                i, j = gene_idx[gene_a], gene_idx[gene_b]
                adj[i][j] = 1.0
                adj[j][i] = 1.0
                matched += 1

    print(f"  Scanned {total_lines:,} interactions")
    print(f"  Added {matched} edges between our {N} genes")

    # --- Step 3: symmetric normalisation D^(-1/2) A D^(-1/2) ---
    adj = symmetric_normalize(adj)

    if cache_path:
        pickle.dump(adj, open(cache_path, 'wb'))
        print(f"Saved adjacency cache to {cache_path}")

    return adj


def build_adj_phenopedia(geneList, weightPhenoPGenes):
    """
    Fallback: connect all Crohn's-associated genes to each other.
    Uses the already-loaded Phenopedia dict — no external download needed.
    """
    N = len(geneList)
    gene_idx = {g: i for i, g in enumerate(geneList)}
    adj = np.eye(N, dtype=np.float32)

    crohn_genes = set(weightPhenoPGenes.keys())
    crohn_in_list = [g for g in geneList if g in crohn_genes]
    print(f"Connecting {len(crohn_in_list)} Crohn's-associated genes (Phenopedia fallback)")

    for i, g1 in enumerate(crohn_in_list):
        for g2 in crohn_in_list[i + 1:]:
            idx1, idx2 = gene_idx[g1], gene_idx[g2]
            adj[idx1][idx2] = 1.0
            adj[idx2][idx1] = 1.0

    return symmetric_normalize(adj)


def symmetric_normalize(adj):
    """D^(-1/2) A D^(-1/2) normalisation."""
    degree = adj.sum(axis=1)
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D = np.diag(d_inv_sqrt)
    return D @ adj @ D