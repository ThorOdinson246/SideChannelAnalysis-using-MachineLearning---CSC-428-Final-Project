# %% [markdown]
# # CSC 428 Final Project: Machine Learning Assisted Side Channel Analysis
#
# **Author:** Your Name
# **Goal:** Implement and compare Random Forest, SVM, and potentially Neural Networks
# for recovering AES key bytes using side-channel traces from the ASCAD dataset.
# **Dataset:** ASCAD Fixed Key (ATMega_AES_v1 - ASCAD.h5)

# %%
# ## 1. Setup and Imports
import numpy as np
import h5py # For reading HDF5 files
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC # LinearSVC can be faster for linear kernel
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

# %% [markdown]
# ## 2. Load ASCAD Dataset
#
# Make sure the `ASCAD.h5` file is accessible in the specified path. This file contains:
# - `traces`: The electromagnetic measurements (50,000 profiling + 10,000 attack).
# - `metadata`: Contains plaintext, key, masks etc. for each trace.
# - `labels`: Often pre-calculated intermediate values (we will calculate our own).

# %%
# Define the path to your ASCAD HDF5 file
ASCAD_FILE_PATH = '../data/ASCAD_desync100.h5' # <--- IMPORTANT: SET THIS PATH

# AES S-box (lookup table used in the SubBytes step)
AES_Sbox = np.array([
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
])
def load_ascad_data(filename):
    """Loads traces, plaintext, and key from the ASCAD HDF5 file."""
    try:
        in_file = h5py.File(filename, "r")
    except FileNotFoundError:
        print(f"Error: File not found at {filename}")
        return None, None, None, None, None
    
    # First, print the file structure to debug
    print("HDF5 file structure:")
    def print_structure(name, obj):
        print(f" - {name} (type: {type(obj).__name__})")
    in_file.visititems(print_structure)
    
    # Try different known ASCAD dataset structures
    try:
        # Structure 1: Original ASCAD format
        if 'traces' in in_file and 'metadata' in in_file:
            print("Using original ASCAD format")
            profiling_traces = np.array(in_file['traces'][:50000, :], dtype=np.int8)
            profiling_plaintext = np.array(in_file['metadata'][:50000]['plaintext'], dtype=np.uint8)
            profiling_key = np.array(in_file['metadata'][:50000]['key'], dtype=np.uint8)
            
            attack_traces = np.array(in_file['traces'][50000:, :], dtype=np.int8)
            attack_plaintext = np.array(in_file['metadata'][50000:]['plaintext'], dtype=np.uint8)
            attack_key = np.array(in_file['metadata'][50000:]['key'], dtype=np.uint8)
        
        # Structure 2: Alternative format with X_profiling/X_attack
        elif 'X_profiling' in in_file and 'X_attack' in in_file:
            print("Using X_profiling/X_attack format")
            profiling_traces = np.array(in_file['X_profiling'][:], dtype=np.int8)
            profiling_plaintext = np.array(in_file['Plaintext_profiling'][:], dtype=np.uint8)
            profiling_key = np.array(in_file['Key_profiling'][:], dtype=np.uint8)
            
            attack_traces = np.array(in_file['X_attack'][:], dtype=np.int8)
            attack_plaintext = np.array(in_file['Plaintext_attack'][:], dtype=np.uint8)
            attack_key = np.array(in_file['Key_attack'][:], dtype=np.uint8)
            
        # Structure 3: Another possible format
        elif 'Profiling_traces' in in_file and 'Attack_traces' in in_file:
            print("Using Profiling_traces/Attack_traces format")
            profiling_traces = np.array(in_file['Profiling_traces']['traces'][:], dtype=np.int8)
            profiling_plaintext = np.array(in_file['Profiling_traces']['metadata']['plaintext'][:], dtype=np.uint8)
            profiling_key = np.array(in_file['Profiling_traces']['metadata']['key'][:], dtype=np.uint8)
            
            attack_traces = np.array(in_file['Attack_traces']['traces'][:], dtype=np.int8)
            attack_plaintext = np.array(in_file['Attack_traces']['metadata']['plaintext'][:], dtype=np.uint8)
            attack_key = np.array(in_file['Attack_traces']['metadata']['key'][:], dtype=np.uint8)
        
        else:
            print("Error: Unknown HDF5 file structure. Please check the dataset format.")
            in_file.close()
            return None, None, None, None, None
            
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        in_file.close()
        return None, None, None, None, None

    # Close the file
    in_file.close()
    
    # Print shapes for debugging
    print(f"Profiling traces shape: {profiling_traces.shape}")
    print(f"Profiling plaintext shape: {profiling_plaintext.shape}")
    
    # Try to verify keys - if they're fixed or different format, handle accordingly
    try:
        # Check if keys are fixed (as expected for this dataset)
        assert np.all(profiling_key == profiling_key[0]), "Error: Keys are not fixed in profiling set."
        assert np.all(attack_key == attack_key[0]), "Error: Keys are not fixed in attack set."
        assert np.all(profiling_key[0] == attack_key[0]), "Error: Profiling key differs from attack key."
        correct_key = profiling_key[0]  # The single fixed key
    except (AssertionError, IndexError) as e:
        print(f"Warning about keys: {str(e)}")
        print("Continuing with first key from profiling set...")
        # Try to get a single key to use anyway
        if len(profiling_key.shape) > 1:
            correct_key = profiling_key[0][0]  # Some datasets have an extra dimension
        else:
            correct_key = profiling_key[0]
        
    return profiling_traces, profiling_plaintext, attack_traces, attack_plaintext, correct_key
# Load the data
profiling_traces, profiling_plaintext, attack_traces, attack_plaintext, correct_key = load_ascad_data(ASCAD_FILE_PATH)

if profiling_traces is not None:
    print("Data Loaded Successfully!")
    print(f"Profiling traces shape: {profiling_traces.shape}") # (50000, 700)
    print(f"Profiling plaintext shape: {profiling_plaintext.shape}") # (50000, 16)
    print(f"Attack traces shape: {attack_traces.shape}") # (10000, 700)
    print(f"Attack plaintext shape: {attack_plaintext.shape}") # (10000, 16)
    print(f"Correct Fixed Key: {correct_key.tolist()}")
else:
    print("Failed to load data. Please check the file path and integrity.")
    # Stop execution if data loading fails
    exit()

# %% [markdown]
# ## 3. Define Target and Preprocess Data
#
# We will target the output of the **first S-box operation** involving the **3rd byte** of the key (index 2) and the **3rd byte** of the plaintext (index 2).
# The intermediate value is `label = Sbox(plaintext[2] ^ key[2])`.
# Our ML models will predict this `label` based on the `profiling_traces`.
#
# **Preprocessing Steps:**
# 1. Calculate labels for the profiling set.
# 2. Split profiling data into training and validation sets.
# 3. Standardize features (traces) using `StandardScaler`.

# %%
TARGET_BYTE_INDEX = 2 # Target the 3rd byte (index 2)

def calculate_labels(plaintexts, key):
    """Calculate the intermediate S-box output for the target byte."""
    return AES_Sbox[plaintexts[:, TARGET_BYTE_INDEX] ^ key[TARGET_BYTE_INDEX]]

# Calculate labels ONLY for the profiling set (we know the key here)
profiling_labels = calculate_labels(profiling_plaintext, correct_key)
print(f"Profiling labels shape: {profiling_labels.shape}") # (50000,)
print(f"Example labels: {profiling_labels[:10]}")

# Split profiling data into training and validation sets (e.g., 80/20 split)
X_profiling, y_profiling = profiling_traces, profiling_labels
X_train, X_val, y_train, y_val = train_test_split(X_profiling, y_profiling, test_size=0.20, random_state=42, stratify=y_profiling)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")

# Feature Scaling (Standardization)
# Fit scaler ONLY on training data, then transform train, validation, and attack data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_attack_scaled = scaler.transform(attack_traces) # Use the same scaler for attack traces

print("Scaling complete.")


# %% [markdown]
# ## 4. Model Training and Evaluation (Classification Task)
#
# We train the models to classify the S-box output (0-255). We evaluate standard classification metrics on the validation set.
#
# **Note:** High classification accuracy doesn't *guarantee* successful key recovery, but it's a good indicator. Key recovery is tested later on the *attack set*.

# %%
# --- Model 1: Random Forest ---
print("\n--- Training Random Forest ---")
start_time = time.time()
rf_model = RandomForestClassifier(n_estimators=100, # Number of trees (adjust as needed)
                                  random_state=42,
                                  n_jobs=-1) # Use all available CPU cores
rf_model.fit(X_train_scaled, y_train)
end_time = time.time()
print(f"RF Training time: {end_time - start_time:.2f} seconds")

# Evaluate on validation set
y_pred_rf = rf_model.predict(X_val_scaled)
acc_rf = accuracy_score(y_val, y_pred_rf)
print(f"RF Validation Accuracy: {acc_rf:.4f}")
print("RF Classification Report (Validation Set):\n", classification_report(y_val, y_pred_rf, zero_division=0))
# print("RF Confusion Matrix:\n", confusion_matrix(y_val, y_pred_rf)) # Can be very large (256x256)


# %%
# --- Model 2: Support Vector Machine (SVM) ---
# Using LinearSVC for potentially faster training with a linear kernel.
# For non-linear kernels (rbf, poly), use SVC, but expect MUCH longer training times.
# If using SVC with non-linear kernel, consider using a smaller subset of data initially.
print("\n--- Training SVM (Linear Kernel) ---")
start_time = time.time()
# svm_model = SVC(kernel='linear', C=1.0, probability=True, random_state=42) # Use probability=True for key recovery using probabilities
svm_model = LinearSVC(C=1.0, random_state=42, max_iter=2000, dual="auto") # Often faster, uses decision_function for ranking
svm_model.fit(X_train_scaled, y_train)
end_time = time.time()
print(f"SVM Training time: {end_time - start_time:.2f} seconds")

# Evaluate on validation set
y_pred_svm = svm_model.predict(X_val_scaled)
acc_svm = accuracy_score(y_val, y_pred_svm)
print(f"SVM Validation Accuracy: {acc_svm:.4f}")
print("SVM Classification Report (Validation Set):\n", classification_report(y_val, y_pred_svm, zero_division=0))


# %%
# --- Model 3: Neural Network (Multi-Layer Perceptron) ---
# As per proposal, this is optional based on time.
print("\n--- Training Neural Network (MLP) ---")
start_time = time.time()
# Simple MLP: 2 hidden layers. Adjust layer sizes and parameters as needed.
nn_model = MLPClassifier(hidden_layer_sizes=(128, 64), # Example architecture
                         activation='relu',
                         solver='adam',
                         batch_size=256,
                         max_iter=50, # Increase if convergence is not reached
                         early_stopping=True, # Stop training if validation score doesn't improve
                         n_iter_no_change=5, # Tolerate 5 epochs with no improvement
                         random_state=42,
                         verbose=True) # Print progress
nn_model.fit(X_train_scaled, y_train)
end_time = time.time()
print(f"NN Training time: {end_time - start_time:.2f} seconds")

# Evaluate on validation set
y_pred_nn = nn_model.predict(X_val_scaled)
acc_nn = accuracy_score(y_val, y_pred_nn)
print(f"NN Validation Accuracy: {acc_nn:.4f}")
print("NN Classification Report (Validation Set):\n", classification_report(y_val, y_pred_nn, zero_division=0))


# %% [markdown]
# ## 5. Key Recovery Attack
#
# This is the crucial step where we use the trained models to predict the key byte on the *attack traces*.
#
# **Strategy:**
# 1. For each attack trace, get the model's prediction probabilities (or decision function scores for SVM) for all 256 possible S-box output values.
# 2. For each *hypothetical key byte* `k_guess` (from 0 to 255):
#    a. Calculate the *hypothesized* S-box output for that trace: `sbox_hyp = Sbox(plaintext[TARGET_BYTE_INDEX] ^ k_guess)`.
#    b. Find the probability/score predicted by the model for this `sbox_hyp`.
# 3. Sum these probabilities/scores across multiple attack traces for each `k_guess`.
# 4. The `k_guess` with the highest total score is the most likely candidate for the correct key byte.
# 5. We measure the "rank" of the correct key byte among all 256 guesses. Success is when the rank is 0 (meaning it's the top guess).
# 6. We plot the rank over an increasing number of attack traces to see how quickly the key is recovered.

# %%
def perform_key_recovery(model, attack_traces_scaled, attack_plaintexts, target_byte_index, correct_key_byte):
    """Performs the key recovery attack and returns the rank of the correct key over traces."""
    
    num_attack_traces = attack_traces_scaled.shape[0]
    key_byte_guesses = range(256)
    
    # Check if the model has predict_proba, otherwise use decision_function (for LinearSVC/SVC)
    if hasattr(model, 'predict_proba'):
        # Get probabilities for all classes for all attack traces
        all_pred_probas = model.predict_proba(attack_traces_scaled)
        # Ensure classes are aligned if model doesn't predict all 256 (rare but possible)
        model_classes = model.classes_
        aligned_probas = np.zeros((num_attack_traces, 256))
        for i, trace_probas in enumerate(all_pred_probas):
           for j, class_label in enumerate(model_classes):
               aligned_probas[i, class_label] = trace_probas[j]
        use_proba = True
    elif hasattr(model, 'decision_function'):
        # Get decision function scores (higher means more confident)
        all_scores = model.decision_function(attack_traces_scaled)
        # Handle binary vs multi-class case from decision_function output shape if needed
        # For multi-class LinearSVC/SVC, shape is (n_samples, n_classes)
        if all_scores.ndim == 1: # Adjust if it's binary and only gives one score column
             print("Warning: Decision function might be for binary. Adapting assuming OvR.")
             # This might need adjustment based on the specific SVM implementation details
             aligned_scores = np.zeros((num_attack_traces, 256))
             aligned_scores[:, 1] = all_scores # Example assumption
             aligned_scores[:, 0] = -all_scores
        else:
             model_classes = model.classes_
             aligned_scores = np.zeros((num_attack_traces, 256))
             for i, trace_scores in enumerate(all_scores):
                for j, class_label in enumerate(model_classes):
                    aligned_scores[i, class_label] = trace_scores[j]
        use_proba = False
    else:
        raise AttributeError("Model must have 'predict_proba' or 'decision_function' method.")

    # Sum of log probabilities (or scores) for each key guess over traces
    sum_log_proba = np.zeros(256)
    # Store rank over increasing number of traces
    ranks_over_traces = np.zeros(num_attack_traces)

    for i in range(num_attack_traces):
        trace_pt = attack_plaintexts[i, target_byte_index]
        
        # Get probabilities/scores for the current trace
        if use_proba:
            trace_predictions = aligned_probas[i]
            # Use log probabilities to avoid underflow, add small epsilon
            log_predictions = np.log(trace_predictions + 1e-40) 
        else:
            trace_predictions = aligned_scores[i]
            # Use scores directly
            log_predictions = trace_predictions 

        for k_guess in key_byte_guesses:
            # Calculate the hypothetical S-box output
            sbox_hyp = AES_Sbox[trace_pt ^ k_guess]
            # Add the log probability/score for this hypothesis
            sum_log_proba[k_guess] += log_predictions[sbox_hyp]

        # Rank the key guesses based on the accumulated log probabilities/scores
        # Higher score/probability is better. argsort gives indices of lowest to highest.
        ranked_guesses = np.argsort(sum_log_proba)[::-1] # [::-1] reverses to highest to lowest

        # Find the rank of the correct key byte (0 is the best rank)
        # np.where returns a tuple, we need the first element (index)
        correct_key_rank = np.where(ranked_guesses == correct_key_byte)[0][0]
        ranks_over_traces[i] = correct_key_rank
        
        # Optional: Print progress periodically
        if (i + 1) % 1000 == 0:
             print(f"Processed {i+1}/{num_attack_traces} traces. Current rank: {correct_key_rank}")

    return ranks_over_traces

# Get the correct key byte for the target index
correct_k_byte = correct_key[TARGET_BYTE_INDEX]
print(f"\nTargeting byte {TARGET_BYTE_INDEX}. Correct value: {correct_k_byte} (0x{correct_k_byte:02x})")

# --- Perform Key Recovery for each model ---
print("\n--- Running Key Recovery Attack (Random Forest) ---")
start_time = time.time()
ranks_rf = perform_key_recovery(rf_model, X_attack_scaled, attack_plaintext, TARGET_BYTE_INDEX, correct_k_byte)
end_time = time.time()
print(f"RF Key Recovery Time: {end_time - start_time:.2f} seconds")

print("\n--- Running Key Recovery Attack (SVM) ---")
start_time = time.time()
ranks_svm = perform_key_recovery(svm_model, X_attack_scaled, attack_plaintext, TARGET_BYTE_INDEX, correct_k_byte)
end_time = time.time()
print(f"SVM Key Recovery Time: {end_time - start_time:.2f} seconds")

# Only run for NN if it was trained
if 'nn_model' in locals():
    print("\n--- Running Key Recovery Attack (Neural Network) ---")
    start_time = time.time()
    ranks_nn = perform_key_recovery(nn_model, X_attack_scaled, attack_plaintext, TARGET_BYTE_INDEX, correct_k_byte)
    end_time = time.time()
    print(f"NN Key Recovery Time: {end_time - start_time:.2f} seconds")
else:
    ranks_nn = None
    print("\nSkipping NN Key Recovery as model was not trained.")

# %% [markdown]
# ## 6. Plot Key Recovery Results
#
# Plot the rank of the correct key byte as a function of the number of attack traces used. A successful attack will show the rank dropping to 0 quickly.

# %%
plt.figure(figsize=(12, 7))

plt.plot(ranks_rf, label='Random Forest', color='blue')
plt.plot(ranks_svm, label='SVM (Linear)', color='red')
if ranks_nn is not None:
    plt.plot(ranks_nn, label='Neural Network (MLP)', color='green')

plt.title(f'Key Recovery Rank for Byte {TARGET_BYTE_INDEX} (Correct Key: 0x{correct_k_byte:02x})')
plt.xlabel('Number of Attack Traces Used')
plt.ylabel('Rank of Correct Key Byte (0 = Best)')
plt.grid(True)
plt.legend()
# Set y-axis limits for better visualization if ranks stay low
# plt.ylim(-1, 50) # Example limit
plt.xlim(0, attack_traces.shape[0]) # Show up to the max number of attack traces
plt.show()

# Print final ranks
print(f"Final Rank (RF) after {attack_traces.shape[0]} traces: {ranks_rf[-1]}")
print(f"Final Rank (SVM) after {attack_traces.shape[0]} traces: {ranks_svm[-1]}")
if ranks_nn is not None:
    print(f"Final Rank (NN) after {attack_traces.shape[0]} traces: {ranks_nn[-1]}")


# %% [markdown]
# ## 7. Conclusion and Discussion
#
# - Summarize the validation accuracies achieved by each model.
# - Compare the key recovery performance based on the plots (how many traces were needed for the rank to reach 0?).
# - Discuss which model performed best for this specific task and dataset.
# - Mention potential improvements (e.g., hyperparameter tuning, different NN architectures like CNNs, feature selection methods beyond scaling, attacking different intermediate values or bytes).
# - Reflect on challenges (e.g., training time, memory usage).

