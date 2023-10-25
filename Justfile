set shell := ["fish", "-c"]

# The folder where the data is stored
export DATA_FOLDER := "data"

# All the data files
data_file1 := "raw_data.csv"
data_file2 := "cleaned_data.csv"

# Path to root folder
root := "/workspaces/debian"

# Path to python executable
python := root / "venv/bin/python"

# Path to project folder
project_path := root / "tesisV4"

# Path to scripts folder
scripts := project_path / "scripts"

clean:
    @echo "Cleaning the file: {{data_file1}}"
    @echo "And saving the result in: {{data_file2}}"
    RAW_DATA_FILE={{ data_file1 }} \
    CLEANED_FILE={{ data_file2 }} \
    {{ python }} {{ scripts }}/clean_data.py

clean-synthetic:
    @echo "Cleaning synthetic data..."
    {{ python }} {{ scripts }}/clean_synthetic_data.py

check:
    @echo "Checking data..."
    {{ python }} {{ scripts }}/check_data.py

analyze:
    @echo "Analyzing data..."
    {{ python }} {{ scripts }}/analyze_data.py