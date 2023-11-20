# Shell Configuration and Project Paths
set shell := ["fish", "-c"]
root_dir := "/home/jorge/Documents/projects"
project_dir := root_dir / "tesisV4"

# Directory Variables
api_dir := project_dir / "api"
etc_dir := project_dir / "etc"
data_dir := project_dir / "data"
venv_dir := project_dir / "venv"
charts_dir := project_dir / "charts"
script_dir := project_dir / "scripts"
results_dir := project_dir / "results"
trained_models_dir := project_dir / "trained_models"

# Executable Variables
python_exec := venv_dir / "bin/python"
uvicorn_exec := venv_dir / "bin/uvicorn"

# Data Files
raw_data_file := "raw_data.csv"
cleaned_data_file := "cleaned_data.csv"

# Synthetic Data Files
raw_synthetic_file := "raw_synthetic_data.csv"
cleaned_synthetic_file := "cleaned_synthetic_data.csv"
gretel_synthetic_file := "data_preview.csv"

# Training and Testing Data Files
train_data_file := "train_data.csv"
val_data_file := "val_data.csv"

# Configuration Files, Scalers, and Model
model_file := "model.pt"
study_file := "study.csv"
scaler_x_file := "scaler_x.pkl"
scaler_y_file := "scaler_y.pkl"
scaler_file := "scaler.pkl"
commands_file := "commands.yaml"

# Synthetic Data Version Configuration
syn_version := "70"
syn_folder_name := "gretel_"
syn_folder := syn_folder_name + syn_version

# Training Variables
debug := "1"
debug_optim := "0"
stopping := "0"
n_trials := "1000"
save_model := "0"

# Chart Variables
charts_data_dir := charts_dir / "data"

# Clean the raw data file to prepare it for analysis
[private]
clean-raw-data:
    @echo "Cleaning the file: {{ raw_data_file }}..."
    @echo "Saving the cleaned result as: {{ cleaned_data_file }}..."
    RAW_DATA_FILE={{ join(data_dir, raw_data_file) }} \
    CLEANED_DATA_FILE={{ join(data_dir, cleaned_data_file) }} \
    {{ python_exec }} {{ script_dir }}/clean_data.py

# Verify the integrity of a data file, checking for non-empty files and NaN values
verify-data-integrity path_to_file:
    @echo "Verifying data in the file: {{ path_to_file }}..."
    FILE_PATH={{ join(data_dir, path_to_file) }} \
    {{ python_exec }} {{ script_dir }}/check_data.py

# Combine multiple data sources and merge them into a single dataset
[private]
combine-and-merge-data:
    @echo "Combining and merging data..."
    @echo "The result will be saved as: {{ raw_data_file }}..."
    RAW_DATA_FILE={{ join(data_dir, raw_data_file) }} \
    CLEANED_DATA_FILE={{ join(data_dir, cleaned_data_file) }} \
    {{ python_exec }} {{ script_dir }}/mix_merge_data.py

# Prepare the directory structure and initial files for synthetic data
[private]
prepare-synthetic-data version:
    @echo "Creating data structure for synthetic data version: {{ version }}..."
    mkdir -p {{ join(data_dir, version) }}
    mkdir -p {{ join(results_dir, version, syn_folder) }}
    mv {{ join(data_dir, gretel_synthetic_file) }} {{ join(data_dir, version, raw_synthetic_file) }}

# Clean the synthetic data to ensure it is ready for model training
[private]
clean-synthetic-data version:
    @echo "Cleaning synthetic data file: {{ raw_synthetic_file }}..."
    @echo "The cleaned data will be saved as: {{ cleaned_synthetic_file }}..."
    ORIGINAL_DATA={{ join(data_dir, cleaned_data_file) }} \
    RAW_SYNTHETIC_DATA={{ join(data_dir, version, raw_synthetic_file) }} \
    CLEANED_SYNTHETIC_DATA={{ join(data_dir, version, cleaned_synthetic_file) }} \
    {{ python_exec }} {{ script_dir }}/clean_synthetic_data.py

# Split the dataset into training and validation sets for model training
[private]
split-data-for-model version:
    @echo "Splitting data into training and validation sets..."
    SYNTHETIC_DATA_PATH={{ join(data_dir, syn_folder, cleaned_synthetic_file) }} \
    TRAIN_DATA_PATH={{ join(data_dir, version, train_data_file) }} \
    VAL_DATA_PATH={{ join(data_dir, version, val_data_file) }} \
    DATA_PATH={{ join(data_dir, cleaned_data_file) }} \
    {{ python_exec }} {{ script_dir }}/mix_split.py

# Set up and verify the synthetic data for model training
setup-model-training-data version:
    @echo "Generating model data for synthetic data version: {{ version }}..."
    @just prepare-synthetic-data {{ version }}
    @just verify-data-integrity {{ join(version, raw_synthetic_file) }}
    @just clean-synthetic-data {{ version }}
    @just split-data-for-model {{ version }}

# Optimize the model's hyperparameters for better performance
optimize-model *args:
    @echo "Optimizing model parameters..."
    DEBUG={{ debug_optim }} N_TRIALS={{ n_trials }} STOPPING={{ stopping }} SAVE_MODEL={{ save_model }} \
    COMMANDS_FILE={{ join(etc_dir, commands_file) }} \
    SCALER_X_FILE={{ join(etc_dir, scaler_x_file) }} \
    SCALER_Y_FILE={{ join(etc_dir, scaler_y_file) }} \
    STUDY_DIR={{ join(results_dir, syn_folder) }} \
    VAL_DATA_PATH={{ join(data_dir, syn_folder, val_data_file) }} \
    TRAIN_DATA_PATH={{ join(data_dir, syn_folder, train_data_file) }} \
    {{ python_exec }} {{ project_dir }}/main.py optimization {{ args }}

# Train the model with the given arguments or default settings
train-model *args:
    @echo "Initiating model training..."
    DEBUG={{ debug }} STOPPING={{ stopping }} SAVE_MODEL={{ save_model }} \
    COMMANDS_FILE={{ join(etc_dir, commands_file) }} \
    SCALER_PATH={{ join(etc_dir, scaler_file) }} \
    VAL_DATA_PATH={{ join(data_dir, syn_folder, val_data_file) }} \
    TRAIN_DATA_PATH={{ join(data_dir, syn_folder, train_data_file) }} \
    {{ python_exec }} {{ project_dir }}/main.py training {{ args }}

# Train the model with custom settings specified by the model's configuration
train-model-with-custom-settings model:
    @echo "Manually training model: {{ model }}..."
    DEBUG={{ debug }} STOPPING={{ stopping }} SAVE_MODEL={{ save_model }} \
    SCALER_X_PATH={{ join(trained_models_dir, model, scaler_x_file) }} \
    SCALER_Y_PATH={{ join(trained_models_dir, model, scaler_y_file) }} \
    MODEL_PATH={{ join(trained_models_dir, model, model_file) }} \
    VAL_DATA_PATH={{ join(data_dir, syn_folder, val_data_file) }} \
    TRAIN_DATA_PATH={{ join(data_dir, syn_folder, train_data_file) }} \
    {{ python_exec }} {{ etc_dir }}/{{ model }}_model.py

# Start the API server with live reloading enabled
start-api:
    @echo "Starting the API with live reloading..."
    SCALER_X_FILE={{ scaler_x_file }} \
    SCALER_Y_FILE={{ scaler_y_file }} \
    MODEL_FILE={{ model_file }} \
    MODELS_DIR={{ trained_models_dir }} \
    {{ uvicorn_exec }} api.server:app --reload

start-deno:
    @echo "Fetching data for inference..."
    DATA_DIR={{ charts_data_dir }} \
    deno run --allow-net --allow-read --allow-write --allow-env charts/main.ts

# Generate the results for the analysis of the effects of number of neurons
generate-neuron-results:
    @echo "Generating results for the analysis of the effects of number of neurons..."
    TRAIN_DATA_PATH={{ join(data_dir, syn_folder, train_data_file) }} \
    {{ python_exec }} {{ etc_dir }}/neuron_results.py