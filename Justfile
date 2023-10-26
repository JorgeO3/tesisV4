# Config of Shell and project paths
set shell := ["fish", "-c"]
root_dir := "/workspaces/debian"
python_exec := root_dir / "venv/bin/python"
project_dir := root_dir / "tesisV4"
script_dir := project_dir / "scripts"
data_dir := project_dir / "data"

# Config of data files
raw_data := "raw_data.csv"
cleaned_data := "cleaned_data.csv"

raw_synthetic := "raw_synthetic_data.csv"
cleaned_synthetic := "cleaned_synthetic_data.csv"
gretel_synthetic := "data_preview.csv"

# The names of the train and test data files
train_data := "train_data.csv"
test_data := "test_data.csv"


# Synthetic data version
sv := "1"

# Train model vars


# Clean the raw data (NOTE: This is not needed anymore, since the data is already clean) 
[private]
clean:
    @echo "Cleaning the file: {{raw_data}}"
    @echo "And saving the result in: {{cleaned_data}}"
    RAW_DATA_FILE={{ join(data_dir, raw_data) }} \
    CLEANED_FILE={{ join(data_dir, cleaned_data) }} \
    {{ python_exec }} {{ script_dir }}/clean_data.py

# The path_to_file variable is used to specify the path to the file to check
# This path starts from the data_dir
# Example: path_to_file = "gretel_70/raw_synthetic_data.csv"
# NOTE: the '70' in the example above is the quality of the synthetic data

# Check if the data is correct (non empty and no NaN values, etc.)
check path_to_file:
    @echo "Checking data..."
    FILE_PATH={{ join(data_dir, path_to_file) }} \
    {{ python_exec }} {{ script_dir }}/check_data.py

# Create the folder structure for the synthetic data
# The n variable is used to specify the folder name of the synthetic data version
[private]
setup-model-data n:
    @echo "Setting up data..."
    mkdir -p {{ join(data_dir, n) }}
    mv {{ join(data_dir, gretel_synthetic) }} {{ join(data_dir, n, raw_synthetic) }}

# Clean the synthetic data
# The n variable is used to specify the folder name of the synthetic data version
[private]
clean-synthetic n:
    @echo "Cleaning synthetic data in file: {{raw_synthetic}}"
    @echo "And saving the result in: {{cleaned_synthetic}}"
    ORIGINAL_DATA={{ join(data_dir, cleaned_data) }} \
    RAW_SYNTHETIC_DATA={{ join(data_dir, n, raw_synthetic) }} \
    CLEANED_SYNTHETIC_DATA={{ join(data_dir, n, cleaned_synthetic) }} \
    {{ python_exec }} {{ script_dir }}/clean_synthetic_data.py

# Split the data into train and test data
# The n variable is used to specify the folder name of the synthetic data version
[private]
split-data n:
    @echo "Splitting data into train and test..."
    DATA_PATH={{ join(data_dir, cleaned_data) }} \
    TRAIN_FILE_PATH={{ join(data_dir, n, train_data) }} \
    TEST_FILE_PATH={{ join(data_dir, n, test_data) }} \
    {{ python_exec }} {{ script_dir }}/split_data.py

# Setups the data for the model; n: synthetic data version
gen-model-data name:
    @echo "Generating data..."
    @just setup-model-data {{ name }}
    @just check {{ join(name, raw_synthetic) }}
    @just clean-synthetic {{ name }}
    @just split-data {{ name }}



