# Therapeutic Reframing Research Pipeline

A comprehensive research pipeline for analyzing semantic paths in therapeutic cognitive reframing using neural activation analysis.

## 🎯 Overview

This project analyzes 520 examples of therapeutic reframing across 13 cognitive pattern types, extracting semantic trajectories through 8 strategic model layers to understand the geometry of therapeutic transformation.

## 📊 Dataset

- **Source**: `data/final/positive_patterns.jsonl`
- **Examples**: 520 (40 per pattern)
- **Pattern Types**: 13 cognitive patterns including:
  - Suicidal Planning & Rationalization
  - Persistent Suicidal Ideation Focus
  - Executive Fatigue & Avolition
  - Self-Critical Rumination
  - Existential Overload & Worthlessness
  - ... and 8 more

- **Text Types Per Example**:
  - **Negative**: `reference_negative_example` (distressed thought)
  - **Transformed**: `reference_transformed_example` (intermediate reframe)
  - **Positive**: `positive_thought_pattern` (healthy thought)

## 🏗️ Pipeline Architecture

### Phase 1: Data Preparation & Extraction
- **1.1**: Parse dataset, extract 3 text types, create train/test splits
- **1.2**: Extract activations from 8 strategic layers using chat template

**Strategic Layers**: `[1, 5, 7, 11, 15, 22, 27, 29]`
- Early (1, 5, 7): Surface features & syntax
- Middle (11, 15): Core semantic transformations
- Late (22, 27, 29): Abstract reasoning & high-level concepts

### Phase 2: Pattern-Specific Path Learning
- **2.1**: Learn 3-landmark paths (negative → transformed → positive)
  - LandmarkPath (piecewise slerp)
  - ParametricCurvePath (Bezier)
  - TangentVectorFieldPath (tangent field)
- **2.2**: Interpret paths at 7 alphas, validate against reference texts

### Phase 3: Universal Pattern Aggregation
- Aggregate paths into universal models using SemanticPathAggregator
- Test 3 methods:
  - `direction_statistics`
  - `curvature_transfer`
  - `relative_geometry`

### Phase 4: Semantic Trajectory Analysis
- Compute geometric properties:
  - **Curvature profiles**: Mean, max, variance per layer
  - **Distance metrics**: Semantic distance, path length, geodesic efficiency
  - **Difficulty scores**: Composite reframing difficulty
  - **Layer importance**: Which layers drive transformation
  - **Landmark accuracy**: How well paths pass through intermediate steps

### Phase 5: Visualizations
- UMAP/t-SNE embeddings (pattern clusters)
- Difficulty heatmaps (pattern × layer)
- Layer importance heatmaps
- Curvature trajectory plots

## 🚀 Usage

### Run Full Pipeline

```bash
.venv/bin/python scripts/therapeutic_reframing/run_all.py
```

Estimated runtime: **6-8 hours**

### Run Individual Phases

```bash
# Phase 1.1: Dataset preparation (~1 minute)
.venv/bin/python scripts/therapeutic_reframing/01_prepare_dataset.py

# Phase 1.2: Activation extraction (~1 hour)
.venv/bin/python scripts/therapeutic_reframing/02_extract_all_activations.py

# Phase 2.1: Path learning (~2-3 hours)
.venv/bin/python scripts/therapeutic_reframing/03_learn_pattern_paths.py

# Phase 2.2: Landmark validation (~1 hour)
.venv/bin/python scripts/therapeutic_reframing/04_extract_landmarks.py

# Phase 3: Universal aggregation (~30 minutes)
.venv/bin/python scripts/therapeutic_reframing/05_learn_universal_patterns.py

# Phase 4: Trajectory analysis (~1-2 hours)
.venv/bin/python scripts/therapeutic_reframing/06_compute_trajectory_properties.py

# Phase 5: Visualizations (~1 hour)
.venv/bin/python scripts/therapeutic_reframing/07_create_visualizations.py
```

## 📁 Output Structure

```
data/therapeutic_reframing/
├── processed/
│   ├── pattern_metadata.csv          # Full dataset
│   ├── train_test_split.json         # 80/20 splits
│   └── strategic_layers.json         # Layer configuration

activations/therapeutic_reframing/
├── by_pattern/
│   └── [13 pattern directories]
│       ├── negative_examples.npz      # (n, 8, hidden_dim)
│       ├── transformed_examples.npz
│       └── positive_examples.npz
└── cache/
    └── activation_index.json

learned_paths/therapeutic_reframing/
├── pattern_specific/
│   └── [13 pattern directories]
│       └── [example_id]_[landmark|parametric|tangent].pkl
└── universal/
    ├── universal_direction_statistics.pkl
    ├── universal_curvature_transfer.pkl
    └── universal_relative_geometry.pkl

analysis/therapeutic_reframing/
├── geometric_analysis/
│   ├── curvature_profiles.csv         # Fine-grained curvature data
│   ├── distance_metrics.csv           # Per-example, per-layer metrics
│   ├── difficulty_scores.csv          # Reframing difficulty
│   ├── layer_importance.csv           # Layer-wise importance
│   └── pattern_summary.csv            # Aggregate statistics
├── interpretation_results/
│   ├── landmark_validation.json       # Interpreted landmarks
│   └── pattern_validation_summary.json
└── visualizations/
    ├── umap_layer_15.png
    ├── umap_by_pattern_layer_15.png
    ├── pattern_difficulty_heatmap.png
    ├── pattern_difficulty_ranking.png
    ├── layer_semantic_distance_heatmap.png
    ├── layer_curvature_heatmap.png
    ├── curvature_trajectories_by_pattern.png
    ├── curvature_by_layer_example.png
    └── difficulty_distribution_by_layer.png
```

## 🔬 Key Research Questions

1. **Which cognitive patterns are hardest to reframe?**
   - See: `pattern_summary.csv`, `pattern_difficulty_ranking.png`

2. **What geometric signatures characterize effective reframing?**
   - See: `curvature_profiles.csv`, `distance_metrics.csv`

3. **Do paths accurately pass through therapeutic landmarks?**
   - See: `landmark_validation.json`

4. **Which model layers drive therapeutic transformation?**
   - See: `layer_importance.csv`, `layer_semantic_distance_heatmap.png`

5. **How do transformations differ across model depth?**
   - See: `difficulty_scores.csv`, `curvature_by_layer_example.png`

6. **Can we learn universal reframing patterns?**
   - See: `universal/` directory, `evaluation_summary.json`

## 📦 Dependencies

```python
torch
transformers
nnsight
numpy
pandas
matplotlib
seaborn
scipy
tqdm
umap-learn  # optional, falls back to sklearn t-SNE
```

## 🔧 Configuration

Edit constants in each script to customize:

- `MODEL_NAME`: Default `"google/gemma-3-4b-it"`
- `STRATEGIC_LAYERS`: Default `[1, 5, 7, 11, 15, 22, 27, 29]`
- `MAX_INTERPRETATION_TOKENS`: Default `60`
- `SAMPLE_SIZE_PER_PATTERN`: Default `5` for validation

## 📝 Notes

- **Chat Template**: Uses proper chat formatting with last token extraction
- **3-Landmark Paths**: Leverages actual intermediate steps from dataset
- **Multi-Layer Analysis**: All paths learned across 8 strategic layers
- **Reproducibility**: Random seeds set (42) for train/test splits

## 🎓 Citation

If you use this pipeline in your research, please cite:

```
[Your citation details here]
```

## 📄 License

[Your license here]

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

## 📧 Contact

[Your contact information]
