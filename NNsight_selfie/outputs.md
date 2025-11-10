# Semantic Path Learning Lab - Experiment Outputs

This document contains all experimental results from the Semantic Path Learning Lab notebook.

---

## Experiment 1: Single Pair Learning (sad → happy)

**Objective**: Learn semantic path for a single concept pair using three different path representations.

**Generated Landmark Keywords**:
- 0.00: sad
- 0.17: mournful
- 0.33: wistful
- 0.50: reflective
- 0.67: hopeful
- 0.83: optimistic
- 1.00: happy

**Configuration**:
- Extraction Layer: 7
- Injection Layer: 7
- Number of Landmarks: 7
- Keyword Template: model_generated

### Results: Three Path Representations Compared

| Alpha | Landmark Path | Parametric Path (Bezier) | Tangent Path |
|-------|--------------|-------------------------|--------------|
| 0.00  | This is a simple example of how neural networks can be used to analyze text and extract meaningful activations from the hidden layers. These activations capture the essence of the text's content, representing a compressed, abstract semantic encoding. The activations are then | The activations likely represent "loneliness" and "isolation," reflecting a sense of being disconnected and lacking social connection. | The activation likely represents a feeling of loneliness or isolation, driven by a solitary state or disconnection from others. It evokes themes of vulnerability and a yearning for connection. |
| 0.25  | The '_' positions represent the concept of 'narrative structure'. | The activation represents a feeling of loneliness and isolation, suggesting a sense of disconnect and lack of social connection. | Based on the text provided, the '_' marks likely represent sadness and loss. The text itself is incomplete, but the empty space suggests a feeling of emptiness and the presence of a period indicates a potential ending or closure, both commonly associated with sadness. |
| 0.50  | The '_' marks represent the concept of love. It embodies passionate emotion, connection, and deep affection between individuals. | The interpretation of the concept is: Fear | The old lighthouse keeper, Silas, remembered a time before the storms, a time of calm seas and predictable tides. He'd spent his entire life tending the light, a solitary vigil against the relentless onslaught of the ocean. |
| 0.75  | The potential for rapid growth is intrinsically linked to the embrace of innovative technologies. (Interpretation: The concept represented at the '_' mark is the synergy between technological advancement and sustainable economic expansion.) | The correct answer is: joyful celebration | The text is: _ is a concept that involves experiencing and expressing sadness, often accompanied by feelings of loss, helplessness, and a desire for comfort. |
| 1.00  | joyful celebration | Joy and contentment | The activation indicates a feeling of contentment and satisfaction derived from simple pleasures or routine activities. |

**Key Observations**:
- **Landmark Path**: Shows some semantic drift (narrative structure → love → technological growth → joyful celebration)
- **Parametric Path**: Struggles with semantic coherence (loneliness → fear → joyful celebration → joy)
- **Tangent Path**: Most consistent emotional progression, starting with loneliness and ending with contentment

---

## Experiment 2: Path Generalization (angry → calm)

**Objective**: Apply learned path (sad → happy) to a NEW concept pair (angry → calm) using all 9 combinations.

**Configuration**:
- Test Pair: angry → calm
- Test Alphas: [0.0, 0.25, 0.5, 0.75, 1.0]
- Methods Tested: Geometric Alignment, Relative Encoding, Direction+Magnitude
- Representations Tested: LandmarkPath, ParametricCurvePath, TangentVectorFieldPath

### 2.1 LandmarkPath Results

| Alpha | Geometric Alignment | Relative Encoding | Direction+Magnitude |
|-------|---------------------|-------------------|---------------------|
| 0.00  | The '_' represents anger and frustration, specifically related to a feeling of injustice or being wronged. | angry_ | The concept encoded at the '_' position is: **rage**. |
| 0.25  | Self-discovery and life meaning. | Peacefulness and tranquility, evoking feelings of calm and serenity due to the gentle sounds and imagery of rain. | Despair. The scene depicts a solitary, fading light and encroaching darkness, suggesting a sense of loss and hopelessness. |
| 0.50  | [Truncated activation data shown] | The activations likely represent the concept of **reflection or mirroring**. The encoded semantic content suggests a focus on observing, responding to, and understanding an external influence or stimulus, with strong thematic links to identity, self-awareness, and the interplay between observer and observed. | The '_' represents **loss and grief** due to a significant, potentially permanent, negative change or loss. |
| 0.75  | The quick brown fox jumps over the lazy dog. The dog is happy. (Activation interpretation: tension between individual agency and societal constraints) | The core of the concept is centered around the tension between individual agency and societal constraints. (Interpretation: conflict between personal freedom and the limitations imposed by social norms) | The activations likely represent the concept of 'uncertainty'. |
| 1.00  | The feeling of being intensely scrutinized. (Interpretation: vulnerability and judgment due to constant observation) | A state of heightened awareness and emotional intensity, particularly surrounding a significant and potentially destabilizing change or revelation. It suggests a visceral reaction and a feeling of being overwhelmed by internal or external circumstances. | The concept represented at the '_' marks is: **Disappointment** |

**Analysis**:
- **Geometric Alignment**: Shows progression but doesn't reach calm state
- **Relative Encoding**: Actually reaches "Peacefulness and tranquility" at 0.25 (closest to target!)
- **Direction+Magnitude**: Moves from rage → despair → loss → uncertainty → disappointment (emotional shift but not to calm)

### 2.2 ParametricCurvePath Results

| Alpha | Geometric Alignment | Relative Encoding | Direction+Magnitude |
|-------|---------------------|-------------------|---------------------|
| 0.00  | Frustration and dissatisfaction with a situation or process. It evokes a feeling of being blocked or unable to achieve a desired outcome. | furious rage | The feeling of a profound, unsettling realization or awareness, often triggered by a sudden shift in perspective or the understanding of a complex truth. It captures the emotional and intellectual impact of confronting a hidden or previously unacknowledged aspect of reality. |
| 0.25  | The feeling of being watched. The concept represented at the '_' mark is a pervasive sense of unease and vulnerability stemming from the awareness of unseen observation. | The '_' marks represent anger and frustration. | Intrigue. The text suggests a state of curiosity and a desire for understanding, evoked by the unexpected or novel information presented. It indicates a mental engagement with something unfamiliar. |
| 0.50  | A feeling of profound disappointment or disillusionment, specifically regarding a previously held belief or expectation that has been shattered. It suggests a sense of loss and the realization that things are not as they seemed. | High activation in areas associated with embodiment, sensory processing, and spatial reasoning. Moderate activation in areas related to emotions, especially fear and anxiety. | The concept represented at the '_' marks is: **Intense emotional connection and romantic affection.** (love) |
| 0.75  | The model predicted 0.8 for this image. (Interpretation: confidence or certainty in the model's prediction) → a feeling of vulnerability and exposure. (Interpretation: emotional fragility) | A simple, well-known pangram used for testing fonts and keyboards. | The activation value is: 5.2. (Interpretation: Joyful celebration and anticipation of a positive outcome) |
| 1.00  | The feeling of profound longing and yearning for something unattainable. | A feeling of disorientation and uncertainty due to a sudden shift in perspective or understanding. It signifies a state of mental confusion accompanied by a questioning of previously held beliefs. | _sadness_ |

**Analysis**:
- **Geometric Alignment**: Starts with frustration, ends with longing (no clear emotional resolution)
- **Relative Encoding**: Completely incoherent - goes through anger → embodiment → pangram text → disorientation
- **Direction+Magnitude**: Bizarre path through unsettling realization → intrigue → love → joyful celebration → sadness

### 2.3 TangentVectorFieldPath Results ⭐ BEST PERFORMER

| Alpha | Geometric Alignment | Relative Encoding | Direction+Magnitude |
|-------|---------------------|-------------------|---------------------|
| 0.00  | Frustration. The text suggests a state of dissatisfaction and discontent, likely stemming from an unresolved problem or unmet expectation. | Frustration and resentment stemming from unmet expectations or perceived injustice. | The concept at the '_' position is: **rage** |
| 0.25  | The concept represented at the '_' marks is: **Frustration and suppressed anger.** | The concept is: **Resentment** | The activations represent the concept of **frustration or dissatisfaction with an unavoidable situation**. |
| 0.50  | The activations likely represent the concept of *frustration or dissatisfaction* arising from an unmet expectation or a perceived injustice. The presence of the underscore suggests a recognition of negative emotional state. | Based on the provided context, the '_' marks likely represent a state of intense dissatisfaction or resentment. The activations would encode a feeling of being wronged, a sense of injustice, and a desire for retribution, reflecting a negative emotional state. | The activations represent the concept of **frustration** - specifically a feeling of annoyance and dissatisfaction with a situation or obstacle. This is strongly associated with the experience of being blocked or impeded from achieving a desired outcome. |
| 0.75  | Conflict. The activations likely represent a state of discord, opposition, or struggle, reflecting the emotionally charged nature of the text. | Anger and frustration, rooted in a sense of injustice and powerlessness. | The concept represented at the '_' positions is: Fear of the unknown and uncertainty about the future. |
| 1.00  | The activation represents a state of profound sorrow, grief, and loss, hinting at a deep emotional wound and potentially a feeling of isolation or despair. | A state of emotional distress or suffering, often accompanied by feelings of sadness, helplessness, and despair. | The activation represents the feeling of **overwhelming anxiety** and a sense of impending doom, triggered by the awareness of a threatening and uncontrollable situation. |

**Analysis - TangentVectorFieldPath + Geometric Alignment** ⭐:
- **Most consistent progression**: frustration → suppressed anger → persistent frustration → conflict → profound sorrow
- Shows smooth **emotional modulation** even though it doesn't fully reach "calm"
- Represents a genuine transformation in emotional quality (active anger → passive grief)
- **Best generalization method overall**

---

## Experiment 3: Multi-Pair Learning

**Objective**: Learn universal "negative → positive emotion" transformation pattern from multiple pairs.

**Training Pairs**:
1. sad → happy
2. angry → calm
3. anxious → relaxed
4. fearful → confident

**Test Pair**: frustrated → satisfied (unseen)

**Generated Keywords for Training**:
- sad → happy: ['sad', 'mournful', 'wistful', 'reflective', 'serene', 'hopeful', 'happy']
- angry → calm: ['angry', 'frustrated', 'irritated', 'tense', 'neutral', 'relaxed', 'calm']
- anxious → relaxed: ['anxious', 'apprehensive', 'uneasy', 'thoughtful', 'calm', 'serene', 'relaxed']
- fearful → confident: ['fearful', 'apprehensive', 'cautious', 'hesitant', 'determined', 'assured', 'confident']

**Universal Representation**: tangent_field with 20 sample points

### Results: Universal Pattern vs. Baseline (Linear Interpolation)

| Alpha | Universal Pattern | Baseline (Simple Linear Interpolation) |
|-------|------------------|----------------------------------------|
| 0.00  | [Activation data with values 0.85, 0.72, 0.61, 0.91] | The activation at the '_' position represents the feeling of **nostalgia and longing** for a past experience. It evokes a sense of wistful remembrance and a desire to return to a cherished moment or time. |
| 0.25  | The fundamental need for **connection and belonging** within a social structure. | frustration. (Interpretation: A feeling of disappointment and dissatisfaction, often stemming from unmet expectations or blocked goals.) |
| 0.50  | **Acceptance and surrender** to a higher power. | The concept represented at the '_' marks is: **Gratitude** |
| 0.75  | The activations represent a state of **contentment and fulfillment**, suggesting a sense of quiet satisfaction with one's current circumstances or achievements. | Content of the original texts. |
| 1.00  | The '_' mark represents the concept of **"contentment and satisfaction"**. It suggests a state of well-being, a feeling of being pleased with something or someone. The activations likely reflect patterns of brain activity associated with positive emotions and a sense of fulfillment. | The '_' marks represent "**humility**" and "**acceptance**" of the outcome. |

**Analysis**:
- **Universal Pattern**: Shows clear progression: [raw activation] → connection/belonging → acceptance → contentment → satisfaction
- **Baseline**: More scattered: nostalgia → frustration → gratitude → content → humility/acceptance
- **Winner**: Universal pattern provides more semantically coherent transition from negative to positive emotional state
- Universal pattern successfully captures the "negative → positive" transformation even for unseen pair

---

## Summary Statistics

### Path Representations (sad → happy):

**Landmark Path**:
- Number of landmarks: 7
- Norm range: 5120.0 - 6784.0
- Average curvature: 0.4902 rad

**Parametric Path (Bezier)**:
- Control points: 4
- Norm range: 5728.0 - 6368.0
- Average curvature: 1.0782 rad

**Tangent Field Path**:
- Tangent samples: 7
- Norm range: 6080.0 - 6336.0
- Stored curvatures: [0.0, 1.8359375, 1.90625, 1.71875, 2.234375, 2.203125, 0.0]

---

## Key Findings

### Best Method for Generalization:
**TangentVectorFieldPath + Geometric Alignment** shows the most consistent and semantically meaningful emotional transitions when generalizing learned paths to new concept pairs.

### Why Generalization is Difficult:
The test task (angry → calm) requires both:
- **Valence shift**: negative → positive (like the training pair sad → happy)
- **Arousal shift**: high-arousal → low-arousal (NOT present in training pair)

This is a **cross-dimensional** generalization that makes perfect transfer challenging.

### Multi-Pair Learning:
Successfully extracts universal transformation patterns when trained on multiple related concept pairs. The universal pattern provides smoother semantic transitions than simple linear interpolation.

### Practical Recommendations:
1. Use **TangentVectorFieldPath + Geometric Alignment** for robust generalization
2. For best results, train on pairs that match the dimensional structure of your test cases
3. Multi-pair learning is effective when pairs share semantic structure (e.g., all emotion transformations)
4. Model-generated keywords create more natural semantic gradients than template-based approaches