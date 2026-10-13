"""Descriptions and definitions that the dataset cards cannot derive from data.

Statistics are recomputed from the CSVs; provenance and meaning are not in them,
so they live here, once, and are written into the cards by
``enrich_dataset_cards.py``.

Every DOI in SOURCES was resolved against Crossref and its title checked against
the work it is meant to name, rather than written from memory.
"""

# Shared vocabulary, written into benchmark_card.json under "definitions" and
# referenced from every card instead of being copied eighteen times.
DEFINITIONS = {
    "cleaning": (
        "Rows dropped between the raw download and the published CSV. A row is "
        "removed when its label is missing or unparseable, when its sequence is "
        "absent, contains non-standard residues that the tokenizers cannot "
        "represent, or is a duplicate of another row under the dataset's key "
        "(protein plus, where applicable, substrate). Deduplication is by key "
        "rather than by whole row, because two measurements of the same enzyme "
        "on the same substrate are the same datapoint even when their metadata "
        "differ."
    ),
    "categorical_high_variance": (
        "A categorical column whose value is nearly unique per row, so it "
        "carries identity rather than category. A column is flagged when its "
        "number of distinct values approaches its number of rows. These columns "
        "are reported because they are never usable as features: a model can "
        "only memorise them, and they can leak the split if the same identifier "
        "reappears across it."
    ),
    "label_summary": (
        "Every card summarises its label in the terms that label is measured "
        "in, chosen by the task's label kind rather than by whether the column "
        "happens to parse as a number. A regression target reports moments; a "
        "binary target reports its positive prevalence, because the median of a "
        "0/1 column states nothing; a per-residue target reports annotated "
        "positions against total residues, because its cells hold position "
        "lists and counting distinct ones counts annotation patterns, roughly "
        "one per protein; and the hierarchical EC target reports classes per "
        "level plus the width of the flattened vector."
    ),
    "unique_classes": (
        "For a categorical column, the number of distinct non-empty values "
        "after whitespace normalisation, counted over the whole published file "
        "rather than per split. For the hierarchical EC label the count is "
        "reported per level, because an EC number is four nested "
        "classifications and a single total would hide that the first level has "
        "seven classes while the fourth has thousands."
    ),
    "ratio_unique_variants": (
        "Number of MMseqs2 clusters divided by number of distinct sequences. It "
        "is near 1 for a set of unrelated proteins and near 0 for a mutational "
        "series around one parent, so it says at a glance which kind of dataset "
        "this is."
    ),
    "mmseqs2_clustering": {
        "note": (
            "MMseqs2 is used for two different purposes at two different "
            "thresholds. Reading one number for the other is the mistake this "
            "entry exists to prevent."
        ),
        "for_similarity_splits": {
            "purpose": (
                "Group sequences so that no cluster is divided between train, "
                "validation and test."
            ),
            "command": "mmseqs cluster <db> <clusterdb> <tmp> --min-seq-id 0.3",
            "min_seq_id": 0.3,
            "other_parameters": (
                "MMseqs2 defaults; the split metadata records every parameter "
                "explicitly, with null meaning the tool's default was used."
            ),
        },
        "for_card_statistics": {
            "purpose": (
                "Describe the internal redundancy of a dataset, which is a "
                "different question from how to split it and uses a stricter "
                "identity and an explicit coverage requirement."
            ),
            "command": (
                "mmseqs cluster <db> <clusterdb> <tmp> --min-seq-id 0.5 "
                "-c 0.8 --cov-mode 0"
            ),
            "min_seq_id": 0.5,
            "coverage": 0.8,
            "cov_mode": 0,
        },
    },
    "standardization": (
        "Regression labels are centred and scaled with the mean and standard "
        "deviation of the training split only, then the same transform is "
        "applied to validation and test. The statistics are recorded in the "
        "split metadata, so the transform is invertible and the original units "
        "recoverable. Fitting on the training split alone is what keeps test "
        "statistics out of the training signal."
    ),
    "split_protocols": (
        "random: proteins assigned 80/10/10 at random, the unit being the "
        "protein so that every measurement of one enzyme lands in one split. "
        "similarity: MMseqs2 clusters at 30% identity assigned whole to a "
        "split, so a test enzyme has no close homolog in training."
    ),
}

# source_name, DOI (Crossref-verified), what the dataset is, what the label is.
SOURCES = {
    "kcat": {
        "source": "CatPred-DB",
        "doi": "10.1038/s41467-025-57215-9",
        "description": (
            "Turnover numbers for enzyme-substrate pairs, assembled by CatPred "
            "from curated kinetics resources. One row is one enzyme measured "
            "against one substrate, so the same protein appears once per "
            "substrate assayed."
        ),
        "label": (
            "k_cat in s^-1, log10-transformed and then standardized on the "
            "training split. The log is applied because turnover spans several "
            "orders of magnitude and an untransformed target would be dominated "
            "by the fastest enzymes."
        ),
    },
    "km": {
        "source": "CatPred-DB",
        "doi": "10.1038/s41467-025-57215-9",
        "description": (
            "Michaelis constants for enzyme-substrate pairs from CatPred-DB. As "
            "with k_cat, one row is one enzyme-substrate measurement."
        ),
        "label": (
            "K_M in mM, log10-transformed and then standardized on the training "
            "split, for the same reason as k_cat."
        ),
    },
    "ph_opt": {
        "source": "BRENDA",
        "doi": "10.1093/nar/gkaa1025",
        "description": (
            "Optimal pH per enzyme, taken from BRENDA. Where an enzyme carries "
            "several reported optima the published value is their median, and "
            "the number of underlying measurements is kept alongside it."
        ),
        "label": (
            "pH optimum in pH units, the median over reported measurements, "
            "standardized on the training split. No log is applied: pH is "
            "already a logarithmic scale."
        ),
    },
    "dtm": {
        "source": "FireProtDB",
        "doi": "10.1093/nar/gkaa981",
        "description": (
            "Change in melting temperature on point mutation, from FireProtDB's "
            "manually curated stability records. One row is one mutant, paired "
            "with its wild-type sequence."
        ),
        "label": (
            "delta T_m in degrees Celsius, mutant minus wild type, standardized "
            "on the training split. Positive means the mutation stabilises."
        ),
    },
    "ddg": {
        "source": "FireProtDB",
        "doi": "10.1093/nar/gkaa981",
        "description": (
            "Change in folding free energy on point mutation, from FireProtDB. "
            "One row is one mutant with its wild-type sequence."
        ),
        "label": (
            "delta delta G in kcal/mol, standardized on the training split. Sign "
            "convention follows the source, where positive is stabilising."
        ),
    },
    "binding": {
        "source": "BioLiP2",
        "doi": "10.1093/nar/gkad630",
        "description": (
            "Whether a protein binds a given small molecule. Positives are "
            "protein-ligand pairs observed in BioLiP2 structures; negatives are "
            "sampled to balance the set, so prevalence is close to 50% by "
            "construction and not by nature. Sequences come from the "
            "crystallised construct, which can be shorter than the full UniProt "
            "entry for the same accession."
        ),
        "label": "Binary: 1 if the pair is an observed complex, 0 otherwise.",
    },
    "ec_number": {
        "source": "UniProt / Swiss-Prot",
        "doi": "10.1093/nar/gkae1010",
        "description": (
            "Enzyme Commission annotation for reviewed Swiss-Prot enzymes. One "
            "row is one protein, which may carry several EC numbers."
        ),
        "label": (
            "Hierarchical multi-label. Each EC string is split on its four "
            "levels and expanded into one binary vector per level, using a "
            "vocabulary fitted on the training split alone. Partial annotations "
            "such as 3.2.2.- contribute only the levels they specify."
        ),
    },
    "active_sites": {
        "source": "UniProt / Swiss-Prot",
        "doi": "10.1093/nar/gkae1010",
        "description": (
            "Catalytic residue positions from Swiss-Prot ACT_SITE annotations. "
            "One row is one protein. Many of these annotations are propagated by "
            "similarity rather than determined experimentally, which is worth "
            "keeping in mind when reading a high score."
        ),
        "label": (
            "Per-residue binary vector as long as the sequence, 1 at an "
            "annotated catalytic position. Positions are 1-indexed in the source "
            "and converted on load. Roughly 0.4% of residues are positive."
        ),
    },
    "binding_site": {
        "source": "UniProt / Swiss-Prot",
        "doi": "10.1093/nar/gkae1010",
        "description": (
            "Ligand-contacting residue positions from Swiss-Prot BINDING "
            "annotations. One row is one protein; the same caveat about "
            "similarity-propagated annotation applies."
        ),
        "label": (
            "Per-residue binary vector, 1 at an annotated binding position. "
            "Roughly 2.5% of residues are positive."
        ),
    },
    "solubility": {
        "source": "ProteinEA solubility benchmark (Ankh)",
        "doi": "10.1101/2023.01.16.524265",
        "description": (
            "Binary solubility labels for the protein set distributed with the "
            "Ankh benchmark suite, retained here so results are comparable with "
            "that literature."
        ),
        "label": "Binary: 1 soluble, 0 insoluble.",
    },
    "align_amylase_activity": {
        "source": "Protein Engineering Tournament (2023 ALIGN round)",
        "doi": "10.1002/prot.70008",
        "description": (
            "Measured activity for an amylase variant library from the Protein "
            "Engineering Tournament. Variants are close relatives of one parent, "
            "so this probes resolution well below the level of enzyme family."
        ),
        "label": (
            "Specific activity in the assay units of the tournament release, "
            "standardized on the training split. One row per variant: repeated "
            "measurements are averaged."
        ),
    },
    "align_amylase_expression": {
        "source": "Protein Engineering Tournament (2023 ALIGN round)",
        "doi": "10.1002/prot.70008",
        "description": (
            "Soluble expression level for the same amylase variant library."
        ),
        "label": (
            "Expression level in the assay units of the tournament release, "
            "standardized on the training split. One row per variant: repeated "
            "measurements are averaged."
        ),
    },
    "align_amylase_stability": {
        "source": "Protein Engineering Tournament (2023 ALIGN round)",
        "doi": "10.1002/prot.70008",
        "description": (
            "Thermostability for the same amylase variant library."
        ),
        "label": (
            "Thermostability in the assay units of the tournament release, "
            "standardized on the training split. One row per variant: repeated "
            "measurements are averaged."
        ),
    },
    "align_glucosidase_activity": {
        "source": "Protein Engineering Tournament (2023 ALIGN round)",
        "doi": "10.1002/prot.70008",
        "description": (
            "Measured activity for a glucosidase variant library from the same "
            "tournament round."
        ),
        "label": (
            "Activity in the assay units of the tournament release, "
            "standardized on the training split. One row per variant: repeated "
            "measurements are averaged."
        ),
    },
    "align_glucosidase_expression": {
        "source": "Protein Engineering Tournament (2023 ALIGN round)",
        "doi": "10.1002/prot.70008",
        "description": "Soluble expression for the glucosidase variant library.",
        "label": (
            "Expression level in the assay units of the tournament release, "
            "standardized on the training split. One row per variant: repeated "
            "measurements are averaged."
        ),
    },
    "align_glucosidase_melting": {
        "source": "Protein Engineering Tournament (2023 ALIGN round)",
        "doi": "10.1002/prot.70008",
        "description": (
            "Melting temperature for the glucosidase variant library. This is "
            "the smallest dataset in the benchmark."
        ),
        "label": (
            "Melting temperature in degrees Celsius, standardized on the "
            "training split. One row per variant: repeated "
            "measurements are averaged."
        ),
    },
    "cleeveland_galox": {
        "source": "Cleveland et al., Fusarium copper radical oxidases",
        "doi": "10.1186/s13068-021-01984-0",
        "description": (
            "Specific activity of fungal copper radical oxidases, the family "
            "galactose oxidase belongs to, measured across a panel of alcohol "
            "and diol substrates. One row is one enzyme-substrate pair, which "
            "makes this the only real-case suite with a ligand axis."
        ),
        "label": (
            "Specific activity in the assay units of the source, standardized on "
            "the training split."
        ),
    },
    "erickson_petase": {
        "source": "Thermotolerant PETase scaffolds from natural diversity",
        "doi": "10.1038/s41467-022-35237-x",
        "description": (
            "PET hydrolase activity for a panel of naturally occurring "
            "scaffolds, measured as aromatic degradation products released, "
            "at five temperatures and six buffer and pH conditions. Temperature, "
            "pH and a one-hot buffer code are model inputs."
        ),
        "provenance_note": (
            "The source table labels its seventh row 30°C, C6 a second time; "
            "it is the 40°C, C6 measurement and is relabelled when cleaning."
        ),
        "label": (
            "Sum of aromatic products in the assay units of the source, "
            "standardized on the training split."
        ),
    },
}
