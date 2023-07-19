import os

import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic

current_dir = os.path.dirname(os.path.abspath(__file__))
mahalanobis_data = os.path.join(current_dir, "../data", "mohalanobis.csv")

df = pd.read_csv(mahalanobis_data)
df = df.sample(frac=1).reset_index(drop=True)

metadata = SingleTableMetadata()

metadata.detect_from_dataframe(data=df)

print(metadata)

synthesizer = CTGANSynthesizer(metadata)
synthesizer.fit(df)

synthetic_data = synthesizer.sample(num_rows=50000)
synthetic_data.to_csv(os.path.join(current_dir, "../data", "synthetic_data.csv"))

quality_report = evaluate_quality(
    df,
    synthetic_data,
    metadata
)

quality_report.get_visualization('Column Shapes')

diagnostic_report = run_diagnostic(
    real_data=df,
    synthetic_data=synthetic_data,
    metadata=metadata)


print(quality_report.get_score())