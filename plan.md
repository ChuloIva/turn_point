## Key Operations to Try

**Transition Direction Vectors:**

- `positive_activation - depressive_activation` = recovery direction
- `resolution_activation - depressive_activation` = therapeutic direction
- `transition_activation - depressive_activation` = change process direction

These difference vectors could capture the semantic "movement" from negative to positive states.

**Reconstruction Experiments:**

- `depressive_activation + (positive - depressive)` should approximate positive states
- `depressive_activation + resolution_direction` might generate transitional language
- Test if `depressive + transition_vector` lands closer to your actual transition samples

## Validation Approaches

**Dot Product Analysis:**

- Check if `transition_activation · (positive - depressive)` is positive (aligned with recovery)
- Measure `cos_similarity(resolution_samples, depressive + transition_vector)`
- See if authentic transitions have higher similarity to computed directions than random text

**Multi-sample Averaging:**

- Average multiple depressive samples to get a "canonical" depressive state
- Same for positive samples and transitions
- These averaged representations might be more stable direction vectors

**Layer-wise Analysis:**

- Earlier layers might capture surface linguistic patterns
- Middle layers could represent emotional content
- Later layers might encode therapeutic concepts
- Try your operations across different layers

## Specific Experiments

1. **Transition Authenticity**: Does `depressive + (resolution - depressive)` produce activations similar to real therapeutic language?
2. **Interpolation Paths**: Create a series `depressive + t*(positive - depressive)` for t ∈ [0,1] and see if intermediate points correspond to transitional states
3. **Orthogonal Decomposition**: Project your transition samples onto the `(positive - depressive)` direction vs orthogonal directions
4. **PCA on Combined Data**: Run PCA on all three types together - do they cluster as expected? Are transitions intermediate?

This could reveal whether the model has learned a consistent geometric representation of psychological recovery processes. The residual stream might encode recovery as literal movement through representation space.

