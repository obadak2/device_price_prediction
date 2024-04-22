import argparse

def get_args():
  """
  Parses command-line arguments for Device Price Prediction script.

  Returns:
      Namespace object containing parsed arguments.
  """

  # Define argument parser
  parser = argparse.ArgumentParser(description="Device Price Prediction")

  # Data paths
  parser.add_argument("--data_path", type=str, required=True,
                      help="Path to the training data CSV file.")

  # Model and preprocessing options (defaults provided)
  parser.add_argument("--model_name", type=str, default="naive",
                      help="Machine learning model name (default: naive)")
  parser.add_argument("--imputation_method", type=str, default="knn",
                      help="Imputation method for missing values (default: knn)")
  parser.add_argument("--scaling_data", action="store_true", default=False,
                      help="Enable scaling of numerical features (default: False)")
  parser.add_argument("--normalize", action="store_true", default=False,
                      help="Enable normalization of numerical features (default: False)")

  # Training and EDA options
  parser.add_argument("--pca", action="store_true", default=False,
                      help="Perform PCA dimensionality reduction (default: False)")

  # Data exploration options (defaults provided)
  parser.add_argument("--imputation", action="store_true", default=True,
                      help="Perform imputation for missing values (default: True)")
  parser.add_argument("--missing_values_per_feature_display", action="store_true", default=True,
                      help="Display missing values per feature (default: True)")
  parser.add_argument("--numerical_dist_display", action="store_true", default=False,
                      help="Display distribution of numerical features (default: False)")
  parser.add_argument("--explore_relash_display", action="store_true", default=False,
                      help="Display exploration of relationships (default: False)")
  parser.add_argument("--corr_display", action="store_true", default=True,
                      help="Display correlation matrix (default: True)")
  
  parser.add_argument("--show_evaluation_metrics", action="store_true", default=False,
                      help="Shows the confusion matrix and printing the metrices of evaluation")

  # Parse arguments
  args = parser.parse_args()

  return args
