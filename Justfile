# Shell Configuration and Project Paths
set shell := ["fish", "-c"]
root_dir := "/workspaces/debian"
python_exec := root_dir / "venv/bin/python"
project_dir := root_dir / "tesisV4"
script_dir := project_dir / "scripts"
data_dir := project_dir / "data"
etc_dir := project_dir / "etc"
results_dir := project_dir / "results"

# Data Files Configuration
raw_data := "raw_data.csv"
cleaned_data := "cleaned_data.csv"

# Synthetic Data
raw_synthetic := "raw_synthetic_data.csv"
cleaned_synthetic := "cleaned_synthetic_data.csv"
gretel_synthetic := "data_preview.csv"

# Training and Testing Data
train_data := "train_data.csv"
test_data := "test_data.csv"

# Configuration Files and Scalers
commands_file := "commands.yaml"
scaler_file := "scaler.pkl"
study_file := "study.csv"

# Synthetic Data Version
syn_version := "70"
syn_folder_name := "gretel_"
syn_folder := syn_folder_name + syn_version

# Variables for Model Training
debug := "0"
n_trials := "500"
save_scaler := "0"
stopping := "0"

# Clean Raw Data (Note: This is no longer necessary as the data is already clean)
[private]
clean:
    @echo "Cleaning the file: {{raw_data}}"
    @echo "And saving the result in: {{cleaned_data}}"
    RAW_DATA_FILE={{ join(data_dir, raw_data) }} \
    CLEANED_FILE={{ join(data_dir, cleaned_data) }} \
    {{ python_exec }} {{ script_dir }}/clean_data.py

# Check Correct Data (non-empty and no NaN values, etc.)
check path_to_file:
    @echo "Checking data..."
    FILE_PATH={{ join(data_dir, path_to_file) }} \
    {{ python_exec }} {{ script_dir }}/check_data.py

# Create Folder Structure for Synthetic Data
[private]
setup-model-data n:
    @echo "Setting up data..."
    mkdir -p {{ join(data_dir, n) }}
    mkdir -p {{ join(results_dir, n, syn_folder) }}
    mv {{ join(data_dir, gretel_synthetic) }} {{ join(data_dir, n, raw_synthetic) }}

# Clean Synthetic Data
[private]
clean-synthetic n:
    @echo "Cleaning synthetic data in file: {{raw_synthetic}}"
    @echo "And saving the result in: {{cleaned_synthetic}}"
    ORIGINAL_DATA={{ join(data_dir, cleaned_data) }} \
    RAW_SYNTHETIC_DATA={{ join(data_dir, n, raw_synthetic) }} \
    CLEANED_SYNTHETIC_DATA={{ join(data_dir, n, cleaned_synthetic) }} \
    {{ python_exec }} {{ script_dir }}/clean_synthetic_data.py

# Split Data into Training and Testing Sets
[private]
split-data n:
    @echo "Splitting data into train and test..."
    DATA_PATH={{ join(data_dir, cleaned_data) }} \
    TRAIN_FILE_PATH={{ join(data_dir, n, train_data) }} \
    TEST_FILE_PATH={{ join(data_dir, n, test_data) }} \
    {{ python_exec }} {{ script_dir }}/split_data.py

# Setup Data for the Model; n: Synthetic Data Version
gen-model-data name:
    @echo "Generating data..."
    @just setup-model-data {{ name }}
    @just check {{ join(name, raw_synthetic) }}
    @just clean-synthetic {{ name }}
    @just split-data {{ name }}

# Optimize the Model
optimize-model *args:
    @echo "Running model..."
    DEBUG={{ debug }} \
    N_TRIALS={{ n_trials }} \
    STOPPING={{ stopping }} \
    SAVE_SCALER={{ save_scaler }} \
    COMMANDS_FILE={{ join(etc_dir, commands_file) }} \
    SCALER_PATH={{ join(etc_dir, scaler_file) }} \
    STUDY_DIR={{ join(results_dir, syn_folder) }} \
    TEST_DATA_PATH={{ join(data_dir, syn_folder, test_data) }} \
    DATA_PATH={{ join(data_dir, syn_folder, train_data) }} \
    SYNTHETIC_DATA_PATH={{ join(data_dir, syn_folder, cleaned_synthetic) }} \
    {{ python_exec }} {{ project_dir }}/main.py {{ args }}

train-model:
    @echo "Training model..."

predict-model:
    @echo "Predicting model..."

test:
    @echo "Hola mundo!"