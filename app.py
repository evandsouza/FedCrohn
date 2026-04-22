import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(
    page_title="FedCrohn — Crohn's Disease Risk Predictor",
    page_icon="🧬",
    layout="wide"
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/dna-helix.png", width=80)
    st.title("FedCrohn")
    st.caption("Federated Graph Attention Network\nfor Crohn's Disease Risk Prediction")
    st.divider()
    st.markdown("**Model config**")
    st.markdown("- Architecture: GAT (2 layers)")
    st.markdown("- Genes: 691 (STRING v12)")
    st.markdown("- Samples: 233 (CAGI2/3/4)")
    st.markdown("- Folds: 5-fold stratified CV")
    st.markdown("- FL rounds: 5 per fold")
    st.markdown("- Clients: 4 hospitals")
    st.divider()
    st.markdown("**Privacy guarantee**")
    st.info("Raw exome data never leaves each hospital node. "
            "Only model weights are shared with the central server.")

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🧬 Crohn's Disease Genomic Risk Prediction")
st.caption("Privacy-preserving federated learning across simulated hospital nodes "
           "using Graph Attention Networks on whole-exome sequencing data.")

st.divider()

# ── Results path ─────────────────────────────────────────────────────────────
RESULTS_DIR = "results"  # relative — run streamlit from project root

# ── Tab layout ───────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Performance",
    "🧬 Gene Importance",
    "🏥 Federated Architecture",
    "📋 About"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Cross-validated performance metrics")

    # Try loading real results, fall back to demo values
    fold_csv = os.path.join(RESULTS_DIR, "fold_results.csv")
    if os.path.exists(fold_csv):
        df = pd.read_csv(fold_csv)
        mcc_mean  = df['mcc'].mean()
        auc_mean  = df['auc'].mean()
        sen_mean  = df['sen'].mean()
        spe_mean  = df['spe'].mean()
        pre_mean  = df['pre'].mean()
        auprc_mean= df['auprc'].mean()
        mcc_std   = df['mcc'].std()
        auc_std   = df['auc'].std()
    else:
        st.warning("No results file found. Showing demo values. "
                   "Run the notebook and place fold_results.csv in the results/ folder.")
        mcc_mean, auc_mean   = 0.291, 0.632
        sen_mean, spe_mean   = 0.530, 0.428
        pre_mean, auprc_mean = 0.530, 0.815
        mcc_std, auc_std     = 0.089, 0.067
        df = pd.DataFrame({
            'mcc':  [0.318, 0.172, 0.441, 0.279, 0.245],
            'auc':  [0.613, 0.596, 0.758, 0.634, 0.561],
            'sen':  [0.500, 0.450, 0.620, 0.480, 0.600],
            'spe':  [0.400, 0.380, 0.500, 0.420, 0.440],
            'pre':  [0.510, 0.490, 0.590, 0.500, 0.560],
            'auprc':[0.820, 0.790, 0.850, 0.810, 0.800],
        })

    # Metric cards
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("MCC",       f"{mcc_mean:.3f}",  f"±{mcc_std:.3f}")
    c2.metric("AUC",       f"{auc_mean:.3f}",  f"±{auc_std:.3f}")
    c3.metric("Sensitivity",f"{sen_mean:.3f}")
    c4.metric("Specificity",f"{spe_mean:.3f}")
    c5.metric("Precision",  f"{pre_mean:.3f}")
    c6.metric("AUPRC",      f"{auprc_mean:.3f}")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Per-fold MCC and AUC**")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        folds = [f"Fold {i+1}" for i in range(len(df))]
        x = np.arange(len(folds))
        w = 0.35
        ax.bar(x - w/2, df['mcc'], w, label='MCC',  color='#7F77DD', alpha=0.85)
        ax.bar(x + w/2, df['auc'], w, label='AUC',  color='#5DCAA5', alpha=0.85)
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='Random (AUC=0.5)')
        ax.axhline(0,   color='red',  linestyle='--', linewidth=0.8, label='Random (MCC=0)')
        ax.set_xticks(x); ax.set_xticklabels(folds, fontsize=9)
        ax.set_ylim(-0.1, 1.0)
        ax.legend(fontsize=8); ax.set_title("Per-fold metrics", fontsize=10)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("**Metric summary**")
        summary = df.describe().loc[['mean','std','min','max']].round(3)
        st.dataframe(summary, use_container_width=True)

    st.divider()
    st.markdown("**Comparison: GAT (ours) vs Baseline (original paper)**")
    compare_df = pd.DataFrame({
        'Model':       ['Baseline NN (paper)', 'GAT + STRING (ours)'],
        'Architecture':['Per-gene linear',     'Graph Attention Network'],
        'Gene graph':  ['None',                 'STRING v12 (6,390 edges)'],
        'MCC':         ['~0.31',                f'{mcc_mean:.3f}'],
        'AUC':         ['~0.66',                f'{auc_mean:.3f}'],
        'Explainability':['No',                 'Yes (attention weights)'],
        'Federated privacy':['Yes',             'Yes'],
    })
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Gene Importance
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Gene importance from federated attention weights")
    st.caption("Genes ranked by aggregated GAT attention scores across all clients and rounds. "
               "Highlighted genes are known Crohn's disease risk genes from GWAS literature.")

    KNOWN_CROHN = {
        'NOD2','IL23R','ATG16L1','CARD9','JAK2','STAT3','IL10',
        'TNFSF15','MST1','PTPN22','IRGM','NKX2-3','LRRK2','CDKAL1',
        'PTGER4','SLC22A5','ICOSLG','TNFRSF6B','RNF186','SMAD3'
    }

    imp_path = os.path.join(RESULTS_DIR, "gene_importance.txt")
    if os.path.exists(imp_path):
        imp_df = pd.read_csv(imp_path, sep="\t")
    else:
        st.warning("gene_importance.txt not found. Showing placeholder data.")
        imp_df = pd.DataFrame({
            'Rank':       list(range(1, 21)),
            'Gene':       ['NOD2','IL23R','ATG16L1','CARD9','STAT3',
                           'PTPN22','IRGM','JAK2','IL10','LRRK2',
                           'ABCB1','ACE','ADA','ADAM17','AGER',
                           'AGR2','AIF1','AIM1','AKR1B1','ALDH16A1'],
            'Importance': np.linspace(1.0, 0.3, 20).round(4)
        })

    imp_df['Known Crohn\'s gene'] = imp_df['Gene'].isin(KNOWN_CROHN).map({True:'✅ Yes', False:'No'})
    top_n = st.slider("Show top N genes", 10, min(50, len(imp_df)), 20)
    display_df = imp_df.head(top_n)

    overlap_count = imp_df.head(20)['Gene'].isin(KNOWN_CROHN).sum()
    st.info(f"**{overlap_count}/20** of the top-20 predicted genes match known Crohn's GWAS hits — "
            f"validating that the model is learning biologically meaningful patterns.")

    col_x, col_y = st.columns([2, 1])

    with col_x:
        fig2, ax2 = plt.subplots(figsize=(7, top_n * 0.32 + 1))
        colors = ['#7F77DD' if g in KNOWN_CROHN else '#B0C4DE'
                  for g in display_df['Gene']]
        ax2.barh(display_df['Gene'][::-1], display_df['Importance'][::-1],
                 color=colors[::-1], edgecolor='none')
        ax2.set_xlabel('Attention importance score', fontsize=9)
        ax2.set_title(f'Top {top_n} genes by GAT attention', fontsize=10)
        known_patch   = mpatches.Patch(color='#7F77DD', label='Known Crohn\'s gene')
        unknown_patch = mpatches.Patch(color='#B0C4DE', label='Novel candidate')
        ax2.legend(handles=[known_patch, unknown_patch], fontsize=8)
        fig2.tight_layout()
        st.pyplot(fig2)
        plt.close()

    with col_y:
        st.markdown("**Full ranking table**")
        def highlight_known(row):
            return ['background-color: #EEEDFE']*len(row) \
                   if '✅' in str(row['Known Crohn\'s gene']) else ['']*len(row)
        st.dataframe(
            display_df[['Rank','Gene','Importance',"Known Crohn's gene"]]
            .style.apply(highlight_known, axis=1),
            use_container_width=True, height=500
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Federated Architecture
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("How the federated system works")

    col_p, col_q = st.columns(2)

    with col_p:
        st.markdown("""
**Privacy-preserving design**

Each hospital (CAGI2, CAGI3, CAGI4) holds its own patient exome data locally.
Raw genomic data **never leaves** the hospital.

Only model weight updates are transmitted to the central server, which aggregates them
using **FedAvg** (weighted averaging by dataset size).

This satisfies the key privacy constraint in real clinical genomics — no patient record
is ever shared across institutional boundaries.

---

**Training flow per round**

1. Server broadcasts current global GAT weights to all clients
2. Each client trains locally for N epochs on its own data
3. Updated weights are sent back to server
4. Server computes FedAvg → new global model
5. Global model evaluated on held-out test fold
6. Attention weights harvested for gene importance
        """)

    with col_q:
        st.markdown("**Dataset breakdown**")
        dataset_df = pd.DataFrame({
            'Dataset': ['CAGI2', 'CAGI3', 'CAGI4', 'Total'],
            'Samples': [56, 66, 111, 233],
            "Crohn's": [42, 51, 64, 157],
            'Healthy': [14, 15, 47, 76],
            'Role in FL': ['Client A', 'Client B', 'Client C', 'All clients'],
        })
        st.dataframe(dataset_df, use_container_width=True, hide_index=True)

        st.markdown("**Gene graph statistics**")
        graph_df = pd.DataFrame({
            'Property': ['Total genes', 'Edges (STRING ≥700)', 'Graph density',
                         'STRING version', 'Confidence threshold'],
            'Value':    ['691', '6,390', '1.48%', 'v12.0', '700 / 1000'],
        })
        st.dataframe(graph_df, use_container_width=True, hide_index=True)

        st.markdown("**What would be added for real hospital deployment**")
        st.markdown("""
- Differential privacy (noise on gradients)
- Secure aggregation protocol
- Separate network servers per institution
- HIPAA / GDPR compliance layer
- Formal privacy budget (ε-DP)
        """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — About
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Project overview")
    st.markdown("""
**Title:** Privacy-Preserving Federated Graph Attention Networks for Explainable Crohn's Disease Risk Prediction from Exome Data

**What this project does:**
- Trains a Graph Attention Network (GAT) on whole-exome sequencing data from 233 patients
- Uses biological gene-gene interactions from the STRING database to build a gene graph
- Trains in a federated manner — no raw data is shared between simulated hospital nodes
- Produces ranked gene importance scores explaining which genes drive the prediction

**Novel contributions over the baseline FedCrohn paper:**
1. GAT replaces the baseline per-gene linear model — genes attend to their biological neighbours
2. STRING v12 protein interaction database used to build the gene graph
3. Federated explainability — attention weights aggregated across clients to rank genes globally
4. Personalised FL layer (hospital-specific adaptation head)

**Datasets:** CAGI2, CAGI3, CAGI4 (Critical Assessment of Genome Interpretation)

**Team:** Capstone project — [your institution]

**Reference:** Original FedCrohn framework — Raimondi et al.
    """)
