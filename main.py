from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset()

dataset.add_goldens_from_csv_file(
    # file_path is the absolute path to your .csv file
    file_path="assets/bench_korean.csv",
    input_col_name="input"
)