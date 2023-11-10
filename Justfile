# Shell Configuration and Project Paths
set shell := ["fish", "-c"]
root_dir := "/home/jorge/Documents/projects/"
project_dir := root_dir / "tesisV4"

api_dir := project_dir / "api"
etc_dir := project_dir / "etc"
data_dir := project_dir / "data"
script_dir := project_dir / "scripts"
results_dir := project_dir / "results"
python_exec := project_dir / "venv/bin/python"
trained_models_dir := project_dir / "trained_models"
e_model_dir := trained_models_dir / "e"
ts_model_dir := trained_models_dir / "ts"
wvp_model_dir := trained_models_dir / "wvp"

# Data Files Configuration
raw_data := "raw_data.csv"
cleaned_data := "cleaned_data.csv"

# Synthetic Data
raw_synthetic := "raw_synthetic_data.csv"
cleaned_synthetic := "cleaned_synthetic_data.csv"
gretel_synthetic := "data_preview.csv"

# Training and Testing Data
train_data := "train_data.csv"
val_data := "val_data.csv"

# Configuration Files, Scalers and Model
model_file := "model.pt"
study_file := "study.csv"
scaler_x := "scaler_x.pkl"
scaler_y := "scaler_y.pkl"
scaler_file := "scaler.pkl"
commands_file := "commands.yaml"

# Synthetic Data Version
syn_version := "70"
syn_folder_name := "gretel_"
syn_folder := syn_folder_name + syn_version

# Variables for Model Training
debug := "1"
stopping := "0"
n_trials := "500"
save_model := "1"

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

[private]
mix-merge:
    @echo "Mixing and merging data..."
    @echo "And saving the result in: {{raw_data}}"
    RAW_DATA_FILE={{ join(data_dir, raw_data) }} \
    CLEANED_FILE={{ join(data_dir, cleaned_data) }} \
    {{ python_exec }} {{ script_dir }}/mix_merge_data.py


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
mix-split n:
    @echo "Splitting data into train and test..."
    SYNTHETIC_DATA_PATH={{ join(data_dir, syn_folder, cleaned_synthetic) }} \
    TRAIN_FILE_PATH={{ join(data_dir, n, train_data) }} \
    TEST_FILE_PATH={{ join(data_dir, n, val_data) }} \
    DATA_PATH={{ join(data_dir, cleaned_data) }} \
    {{ python_exec }} {{ script_dir }}/mix_split.py

# Setup Data for the Model; n: Synthetic Data Version
gen-model-data name:
    @echo "Generating data..."
    @just setup-model-data {{ name }}
    @just check {{ join(name, raw_synthetic) }}
    @just clean-synthetic {{ name }}
    @just mix-split {{ name }}

# Optimize the Model
optimize-model *args:
    @echo "Optimizing model..."
    DEBUG={{ debug }} \
    N_TRIALS={{ n_trials }} \
    STOPPING={{ stopping }} \
    SAVE_MODEL={{ save_model }} \
    COMMANDS_FILE={{ join(etc_dir, commands_file) }} \
    SCALER_X={{ join(etc_dir, scaler_x) }} \
    SCALER_X={{ join(etc_dir, scaler_y) }} \
    STUDY_DIR={{ join(results_dir, syn_folder) }} \
    VAL_DATA_PATH={{ join(data_dir, syn_folder, val_data) }} \
    TRAIN_DATA_PATH={{ join(data_dir, syn_folder, train_data) }} \
    {{ python_exec }} {{ project_dir }}/main.py optimization {{ args }}

# Manual model training
train-model *args:
    @echo "Training model..."
    DEBUG={{ debug }} \
    STOPPING={{ stopping }} \
    SAVE_MODEL={{ save_model }} \
    COMMANDS_FILE={{ join(etc_dir, commands_file) }} \
    SCALER_PATH={{ join(etc_dir, scaler_file) }} \
    val_data_PATH={{ join(data_dir, syn_folder, val_data) }} \
    TRAIN_DATA_PATH={{ join(data_dir, syn_folder, train_data) }} \
    {{ python_exec }} {{ project_dir }}/main.py training {{ args }}

predict-model:
    @echo "Predicting model..."


train-manual-model model:
    DEBUG={{ debug }} \
    STOPPING={{ stopping }} \
    SAVE_MODEL={{ save_model }} \
    SCALER_X={{ join(trained_models_dir, model, scaler_x) }} \
    SCALER_Y={{ join(trained_models_dir, model, scaler_y) }} \
    MODEL_PATH={{ join(trained_models_dir, model, model_file) }} \
    VAL_DATA_PATH={{ join(data_dir, syn_folder, val_data) }} \
    TRAIN_DATA_PATH={{ join(data_dir, syn_folder, train_data) }} \
    {{ python_exec }} {{ etc_dir }}/{{ model }}_model.py

start-api:
    @echo "Starting API..."
    SCALER_X={{ scaler_x }} \
    SCALER_Y={{ scaler_y }} \
    MODEL_FILE={{ model_file }} \
    MODELS_DIR={{ trained_models_dir }} \
    {{ python_exec }} {{ api_dir }}/server.py