from BaseClass.BaseClass import DevicePricePredictor
from sklearn.model_selection import train_test_split
from fastapi import FastAPI, Body
from argparser import get_args
from typing import Dict
import numpy as np
import uvicorn

def create_fastapi_app(device_model: DevicePricePredictor) -> FastAPI:
  app = FastAPI()

  @app.post("/predict_price")
  def predict(data: Dict = Body(...)) -> Dict:
    features_list = list(data.values())
    features_array = np.zeros(len(features_list))
    for i in range(len(features_list)):
       features_array[i] = features_list[i] 
    features_array = device_model.preprocess_data(data=features_array,
                                       scaling_data=args.scaling_data,
                                       normalize=args.normalize,
                                       train=False,
                                       pca=args.pca,
                                       imputation_method=args.imputation_method,
                                       imputation=False)
    predicted_price = device_model.predict_price(features_array)

    return {"predicted_price": predicted_price.tolist()}

  return app

def load_and_preprocess_data(args, device_model):
  data = device_model.load_data(args.data_path)
  y_train = data['target'].values
  X_train = data.drop('target', axis=1)
  X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, stratify=y_train, random_state=42)

  X_train = device_model.preprocess_data(data=X_train,
                                       scaling_data=args.scaling_data,
                                       normalize=args.normalize,
                                       train=True,
                                       pca=args.pca,
                                       imputation_method=args.imputation_method,
                                       imputation=args.imputation)
  if not isinstance(X_train, np.ndarray):
    X_train = X_train.values
  device_model.train_model(X_train, y_train)

  X_test = device_model.preprocess_data(data=X_test,
                                       scaling_data=args.scaling_data,
                                       normalize=args.normalize,
                                       train=False,
                                       pca=args.pca,
                                       imputation_method=args.imputation_method,
                                       imputation=args.imputation)
  if not isinstance(X_train, np.ndarray):
    X_test = X_test.values
  if args.show_evaluation_metrics:
    device_model.evaluate_model(X_test, y_test)

  return X_train, X_test, y_train, y_test

def main(args):
    device_model = DevicePricePredictor(args.model_name)
    X_train, X_test, y_train, y_test = load_and_preprocess_data(args, device_model)
    device_model.train_model(X_train, y_train)
    app = create_fastapi_app(device_model=device_model)
    return app
    
if __name__ == "__main__":
    args = get_args()
    app = main(args=args)
    uvicorn.run(app, host="localhost", port=8000)

  