#
# --- Imports and Initial Setup ---
#

import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from repeng import ControlVector, ControlModel, DatasetEntry
from nnsight import LanguageModel
import re
from typing import List, Dict
from dataclasses import dataclass
import copy
import gc
import random

#
# --- Configuration ---
#

# Model and Tokenizer Configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
TOKENIZER.pad_token_id = 0

# Device Configuration (prioritizes MPS, then CUDA, then CPU)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")

# Steering and Chat Configuration
USER_TAG, ASST_TAG = "[INST]", "[/INST]"
STEERING_LAYERS = list(range(-5, -18, -1)) # Layers to apply steering vectors

#
# --- Data Structures for Activation Capture ---
#

@dataclass
class Layer16ActivationCapture:
    """Stores Layer 16 activations from a specific point in generation."""
    token_position: int
    layer_16_activation: torch.Tensor
    steering_strength: float
    token_text: str
    generation_step: str

@dataclass
class Layer16TransitionActivationSet:
    """A complete set of activations captured during a single sentiment transition."""
    question: str
    baseline_activations: Layer16ActivationCapture
    negative_activations: Layer16ActivationCapture
    transition_start: Layer16ActivationCapture
    transition_mid: Layer16ActivationCapture
    transition_end: Layer16ActivationCapture
    full_response: str
    transition_tokens: List[str]
    control_vector: ControlVector
    steering_layers: List[int]

#
# --- Core Memory and Model Management ---
#

# Global variables to hold model instances
current_repeng_model = None
current_control_model = None
current_nnsight_model = None

def load_repeng_model():
    """Loads the model and wraps it with RepEng's ControlModel."""
    global current_repeng_model, current_control_model
    if current_repeng_model is not None:
        return

    print("Loading RepEng model...")
    model_args = {'torch_dtype': torch.float16}
    if DEVICE.type == "mps":
        # Load on CPU first, then move to MPS to manage memory
        model_args['device_map'] = None
        current_repeng_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_args)
        current_repeng_model = current_repeng_model.to(DEVICE)
    else:
        model_args['device_map'] = "auto"
        current_repeng_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_args)

    current_control_model = ControlModel(current_repeng_model, STEERING_LAYERS)
    print(f"RepEng model loaded on {current_repeng_model.device}.")

def unload_model():
    """Unloads any active model and clears GPU cache to free memory."""
    global current_repeng_model, current_control_model, current_nnsight_model
    
    if current_repeng_model is not None or current_nnsight_model is not None:
        print("Unloading model and clearing cache...")
        del current_repeng_model
        del current_control_model
        del current_nnsight_model
        current_repeng_model, current_control_model, current_nnsight_model = None, None, None
        
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        elif DEVICE.type == "mps":
            torch.mps.empty_cache()
        print("Model unloaded and memory freed.")

def load_nnsight_model():
    """Loads the model with the NNsight wrapper for activation capture."""
    global current_nnsight_model
    if current_nnsight_model is not None:
        return

    print("Loading NNsight model...")
    current_nnsight_model = LanguageModel(
        MODEL_NAME,
        torch_dtype=torch.float16,
        dispatch=True,
        device_map=DEVICE
    )
    print("NNsight model loaded.")

#
# --- Activation Capture and Analysis Logic ---
#

def extract_layer_16_steering_component(control_vector, strength, shape, device):
    """Calculates the precise steering vector applied to Layer 16."""
    if 16 not in control_vector.directions:
        return torch.zeros(shape, dtype=torch.float16, device=device)
    
    direction = torch.tensor(
        strength * control_vector.directions[16],
        device=device,
        dtype=torch.float16
    ).reshape(1, 1, -1)
    
    return direction.expand(shape)

def subtract_layer_16_steering(steered_activation, control_vector, strength, device):
    """Removes the steering component to get the 'clean' activation."""
    steering_component = extract_layer_16_steering_component(
        control_vector, strength, steered_activation.shape, device
    )
    return steered_activation - steering_component

def capture_layer_16_activations(model, text, position=-1):
    """Uses NNsight to capture Layer 16 activations at a specific token position."""
    with model.trace(text) as tracer:
        # The output of the 16th layer's forward pass (residual stream)
        hidden_states = model.model.layers[16].output[0]
        
        # Save the activation at the specified token position
        if len(hidden_states.shape) == 3: # (batch, seq_len, hidden_dim)
            activation = hidden_states[:, position, :].save()
        else: # (seq_len, hidden_dim)
            activation = hidden_states[position, :].save()
            
    return activation

#
# --- Data Preparation ---
#

def load_cognitive_questions(filepath: str) -> List[str]:
    """Loads and flattens questions from a markdown file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print("Warning: cognitive_pattern_questions.md not found. Using a default question.")
        return ["How do you feel when you think about your future?"]

    questions = []
    for line in content.split('\n'):
        if re.match(r'^\d+\. ', line):
            question = re.sub(r'^\d+\\. ', '', line).strip()
            if question:
                questions.append(question)
    return questions

def create_control_dataset() -> List[DatasetEntry]:
    """Creates the dataset for training the RepEng control vector."""
    # This is a fallback based on the original notebook's data.
    # For better results, use the original `all_truncated_outputs.json` from RepEng.
    suffixes = [
        "I feel that", "I think", "It seems to me", "My opinion is",
        "From my perspective", "I believe", "What I mean is", "To be honest"
    ]
    
    dataset = []
    positive_persona, negative_persona = "happy", "sad"
    
    for suffix in suffixes:
        positive_template = f"{USER_TAG} Act as if you're extremely {positive_persona}. {ASST_TAG} {suffix}"
        negative_template = f"{USER_TAG} Act as if you're extremely {negative_persona}. {ASST_TAG} {suffix}"
        dataset.append(DatasetEntry(positive=positive_template, negative=negative_template))
        
    return dataset

#
# --- 3-Phase Workflow Implementation ---
#

def phase_1_train_steering_vectors(control_dataset):
    """Phase 1: Load RepEng model, train the control vector, and unload."""
    print("\n--- Starting Phase 1: Training Steering Vectors ---")
    load_repeng_model()
    
    print("Training control vector...")
    control_vector = ControlVector.train(
        current_control_model,
        TOKENIZER,
        control_dataset,
        method="pca_center",
        batch_size=4 
    )
    
    print("Control vector training complete.")
    unload_model()
    print("--- Phase 1 Complete ---")
    return control_vector

def phase_2_generate_steering_responses(questions, control_vector, num_samples):
    """Phase 2: Load RepEng, generate steered responses, and unload."""
    print("\n--- Starting Phase 2: Generating Steered Responses ---")
    load_repeng_model()
    
    selected_questions = random.sample(questions, min(num_samples, len(questions)))
    response_data_list = []
    
    for i, question in enumerate(selected_questions):
        print(f"Generating response for question {i+1}/{len(selected_questions)}...")
        input_text = f"{USER_TAG} {question} {ASST_TAG}"
        
        # Generate the initial negative part of the response
        current_control_model.set_control(control_vector, -2.0)
        input_ids = TOKENIZER(input_text, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            negative_output = current_control_model.generate(
                **input_ids,
                max_new_tokens=30, # Shorter negative portion
                do_sample=True,
                temperature=0.7,
                pad_token_id=TOKENIZER.eos_token_id
            )
        
        full_negative_text = TOKENIZER.decode(negative_output.squeeze(), skip_special_tokens=True)
        negative_response = full_negative_text.split(ASST_TAG)[-1].strip()
        
        # Create the prompt for the positive continuation
        continuation_prompt = f"{input_text} {negative_response}"
        
        # Set a positive steering vector and generate the rest
        current_control_model.set_control(control_vector, 1.8)
        continuation_ids = TOKENIZER(continuation_prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            positive_output = current_control_model.generate(
                **continuation_ids,
                max_new_tokens=50, # Longer positive portion
                do_sample=True,
                temperature=0.8,
                pad_token_id=TOKENIZER.eos_token_id
            )

        full_positive_text = TOKENIZER.decode(positive_output.squeeze(), skip_special_tokens=True)
        
        response_data = {
            'question': question,
            'input_text': input_text,
            'full_response_text': full_positive_text,
        }
        response_data_list.append(response_data)

    print(f"Generated {len(response_data_list)} responses.")
    unload_model()
    print("--- Phase 2 Complete ---")
    return response_data_list

def phase_3_capture_activations(response_data_list, control_vector):
    """Phase 3: Load NNsight, capture activations from generated text, and unload."""
    print("\n--- Starting Phase 3: Capturing Activations ---")
    load_nnsight_model()
    
    activation_sets = []
    
    for i, data in enumerate(response_data_list):
        print(f"Capturing activations for response {i+1}/{len(response_data_list)}...")
        
        full_text = data['full_response_text']
        input_ids = TOKENIZER.encode(full_text, return_tensors='pt')[0]
        input_tokens = len(TOKENIZER.encode(data['input_text']))
        
        # Define capture points
        baseline_pos = input_tokens - 1
        neg_pos = input_tokens + 15 # Approx. middle of negative part
        trans_start_pos = input_tokens + 30 # Start of positive part
        trans_end_pos = len(input_ids) -1 # End of response
        
        # 1. Baseline activation
        baseline_act = capture_layer_16_activations(current_nnsight_model, full_text, baseline_pos)
        
        # 2. Negative activation (clean)
        neg_steered_act = capture_layer_16_activations(current_nnsight_model, full_text, neg_pos)
        neg_clean_act = subtract_layer_16_steering(neg_steered_act, control_vector, -2.0, DEVICE)

        # 3. Transition start activation (clean)
        start_steered_act = capture_layer_16_activations(current_nnsight_model, full_text, trans_start_pos)
        start_clean_act = subtract_layer_16_steering(start_steered_act, control_vector, 1.8, DEVICE)
        
        # 4. Transition end activation (clean)
        end_steered_act = capture_layer_16_activations(current_nnsight_model, full_text, trans_end_pos)
        end_clean_act = subtract_layer_16_steering(end_steered_act, control_vector, 1.8, DEVICE)
        
        # Dummy capture for transition mid-point for structure compatibility
        mid_capture = Layer16ActivationCapture(
            token_position=-1, layer_16_activation=torch.zeros_like(end_clean_act), 
            steering_strength=1.8, token_text="[MID-DUMMY]", generation_step="transition_mid"
        )
        
        # Assemble the full activation set
        activation_set = Layer16TransitionActivationSet(
            question=data['question'],
            baseline_activations=Layer16ActivationCapture(baseline_pos, baseline_act, 0.0, "[BASELINE]", "baseline"),
            negative_activations=Layer16ActivationCapture(neg_pos, neg_clean_act, -2.0, "[NEGATIVE]", "negative"),
            transition_start=Layer16ActivationCapture(trans_start_pos, start_clean_act, 1.8, "[START]", "transition_start"),
            transition_mid=mid_capture,
            transition_end=Layer16ActivationCapture(trans_end_pos, end_clean_act, 1.8, "[END]", "transition_end"),
            full_response=full_text,
            transition_tokens=[], # Simplified; can be re-tokenized if needed
            control_vector=control_vector,
            steering_layers=[32 + l for l in STEERING_LAYERS]
        )
        activation_sets.append(activation_set)
        
    print(f"Captured {len(activation_sets)} activation sets.")
    unload_model()
    print("--- Phase 3 Complete ---")
    return activation_sets

def run_complete_workflow(questions_file, num_samples=3):
    """Executes the full 3-phase workflow."""
    print("======================================================")
    print("Starting Memory-Efficient Sentiment Transition Capture")
    print("======================================================")
    
    # Data Loading
    control_dataset = create_control_dataset()
    all_questions = load_cognitive_questions(questions_file)
    
    # Phase 1
    control_vector = phase_1_train_steering_vectors(control_dataset)
    
    # Phase 2
    response_data_list = phase_2_generate_steering_responses(all_questions, control_vector, num_samples)
    
    # Phase 3
    final_activation_sets = phase_3_capture_activations(response_data_list, control_vector)
    
    print("\n======================================================")
    print("Workflow Finished Successfully")
    print("======================================================")
    
    return final_activation_sets

#
# --- Main Execution ---
#
if __name__ == '__main__':
    # Execute the workflow for 2 samples
    # Make sure you have a 'cognitive_pattern_questions.md' file in the same directory.
    activation_results = run_complete_workflow('cognitive_pattern_questions.md', num_samples=2)

    #
    # --- Analysis of Results ---
    #
    if activation_results:
        print(f"\n--- Analysis of {len(activation_results)} Result(s) ---")
        for i, result in enumerate(activation_results):
            print(f"\n--- Result {i+1} ---")
            print(f"Question: {result.question}")
            print(f"Full Response: {result.full_response.split(ASST_TAG)[-1].strip()[:150]}...")
            
            # Compare activation magnitudes
            baseline_mag = torch.norm(result.baseline_activations.layer_16_activation).item()
            negative_mag = torch.norm(result.negative_activations.layer_16_activation).item()
            positive_mag = torch.norm(result.transition_end.layer_16_activation).item()
            
            print(f"  - Baseline L16 Magnitude: {baseline_mag:.2f}")
            print(f"  - Clean Negative L16 Magnitude: {negative_mag:.2f}")
            print(f"  - Clean Positive L16 Magnitude: {positive_mag:.2f}")