from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, precision_score, roc_auc_score, accuracy_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import IterativeImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn import preprocessing
import matplotlib.pyplot as plt
from typing import Union
from sklearn.svm import SVC
import seaborn as sb 
import pandas as pd
import numpy as np

class DevicePricePredictor:
  def __init__(self, model_name:str):
    self.model_name = model_name
    self.features = []
    self.model = None

  def load_data(self, data_path: str) -> pd.DataFrame:
    """
      This function is used to load a csv data and gave an assumption that the last
      column is the labels column and change it's name to be taget
      Argumets:
        data_path(str):path to the dataset
      Returns:
        data(pd.DataFrame):loaded csv dataframe after changing the labels column name to target 
    """
    data = pd.read_csv(data_path)
    columns = list(data.columns)
    columns[-1] = 'target'
    data.columns = columns
    return data

  def perform_imputation(self,
                          data:Union[np.ndarray, pd.DataFrame]=None,
                          imputation_method:str='knn') -> pd.DataFrame:
    """
      This function is used to perform imputation on the passed dataset based on the imputation method decided
      Argumets:
        data(Union[np.ndarray, pd.DataFrame]):the dataset
        imputation_method(str):the imputation method that need to be applied
      Returns:
        data(pd.DataFrame):loaded csv dataframe after changing the labels column name to target 
    """
    if not isinstance(data, pd.DataFrame):
      data = pd.DataFrame.from_records(data)

    if imputation_method =='mice':
      mice_imputer = IterativeImputer(random_state=100, max_iter=5)
      df_mice_imputed = mice_imputer.fit_transform(data)

      df_mice_imputed = pd.DataFrame(df_mice_imputed, columns=data.columns, index=data.index)
      data = df_mice_imputed

    elif imputation_method =='knn':
      imputer = KNNImputer(n_neighbors=5)
      filled_data = imputer.fit_transform(data)

      df_imputed = pd.DataFrame(filled_data, columns=data.columns, index=data.index)
      data = df_imputed

    return data

  def perform_eda(self,
                  data:Union[np.ndarray, pd.DataFrame]=None,
                  missing_values_per_feature_display:bool=False,
                  numerical_dist_display:bool=False,
                  explore_relash_display:bool=False,
                  corr_display:bool=True) -> None:
    """
      This function is used to perform EDA methods on the data
      Argumets:
        data(Union[np.ndarray, pd.DataFrame]):the dataset
        missing_values_per_feature_display(bool):boolean variable to decide whether to show missed values per features or not
        numerical_dist_display(bool):boolean variable to explore distribution of numerical features with histograms
        explore_relash_display(bool):boolean variable to explore relationships with scatter plots
        corr_display(bool):boolean variable to explore relationships between features using heatmap
      Returns:

    """

    if not isinstance(data, pd.DataFrame):
      data = pd.DataFrame(data)
    
    if missing_values_per_feature_display:
      # Create a bar chart to visualize missing values
      data.isnull().sum().plot(kind='bar')
      plt.title("Missing Values per Feature")
      plt.xlabel("Feature")
      plt.ylabel("Count of Missing Values")
      plt.show()
    
    if numerical_dist_display:
      # Explore distribution of numerical features with histograms
      for col in data.columns:
        if data[col].dtype != object:
          sb.histplot(data=data, x=col)
          plt.show()

    if explore_relash_display:
      # Explore relationships with scatter plots (consider using pairplots for many features)
      sb.scatterplot(x="battery_pe", y="price", data=data)
      plt.show()

    if corr_display:
      plt.figure(figsize=(20, 10))
      # plotting correlation heatmap 
      dataplot = sb.heatmap(data.corr(), cmap="YlGnBu", annot=True)
      plt.show() 



  def preprocess_data(self,
                      scaling_data:bool=False,
                      normalize:bool=False,
                      imputation:bool=False,
                      imputation_method:str='knn',
                      train:bool=True,
                      pca:bool=False,
                      data:Union[np.ndarray, pd.DataFrame]=None) -> Union[np.ndarray, pd.DataFrame]:
    
    """
      This function is used to perform preprocessing on the data
      Argumets:
        data(Union[np.ndarray, pd.DataFrame]):the dataset
        scaling_data(bool):boolean variable to decide whether to apply standardization or not
        normalize(bool):boolean variable to decide whether to apply normalization or not
        imputation(bool):boolean variable to decide whether to apply imputation or not
        pca(bool):boolean variable to decide whether to apply pca or not
        train(bool):boolean variable to decide whether the current data is training or not
        imputation_method(str):the imputation method name can be (knn, mice)
      Returns:
        data(Union[np.ndarray, pd.DataFrame])
    """

    if imputation:
      data = self.perform_imputation(data, imputation_method)


    if scaling_data:
      if train:
        self.scaler = StandardScaler().fit(data)
        data = self.scaler.transform(data)
      else:
        data = self.scaler.transform(data)

    if normalize:
      if train:
        self.scaler = preprocessing.Normalizer().fit(data)
        data = self.scaler.transform(data)
      else:
        data = self.scaler.transform(data)
    if pca:
      pca = PCA(n_components=0.95)  # Reduce to 95% variance
      data = pca.fit_transform(data)
    return data
    
        

  def train_model(self, X_train, y_train) -> None:
    """
      This function is used to create and fit the model chosen before
      Argumets:
        X_train(np.ndarray):the features
        y_train(np.ndarray):the labels
      Returns:
    """
    if self.model_name=='rf':
      self.model = RandomForestClassifier()
    elif self.model_name=='lr':
      self.model = LogisticRegression(random_state=42)
    elif self.model_name=='naive':
      self.model = GaussianNB()
    elif self.model_name=='svm':
      self.model = SVC(gamma='auto')
    elif self.model_name=='lir':
      self.model = LinearRegression()


    self.model.fit(X_train, y_train)

  def evaluate_model(self, X_test: np.ndarray[np.ndarray], y_test: np.ndarray) -> None:
      """
      This function is used to evaluated the fitted model
      Argumets:
        X_test(np.ndarray):the features
        y_test(np.ndarray):the labels
      Returns:
      """
      # Predict on test data
      y_pred = self.model.predict(X_test)

      # Calculate metrics (assuming target variable is discretized into price categories)
      print("Classification Report:")
      print(classification_report(y_test, y_pred))

      # Confusion Matrix (assuming target variable is discretized into price categories)
      cm = confusion_matrix(y_test, y_pred)
      print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
      print("Recall:", recall_score(y_test, y_pred, average='weighted'))
      print("Precision:", precision_score(y_test, y_pred, average='weighted'))
      print("Accuracy:", accuracy_score(y_test, y_pred))
      print('------------------------------------------')

      self.plot_confusion_matrix(cm)


  def predict_price(self, X_test:np.ndarray[np.ndarray]=None) -> np.ndarray:
    """
    This function is used to predict using the fitted model
    Argumets:
      X_test(np.ndarray):the features
      y_test(np.ndarray):the labels
    Returns:
      y_pred(np.ndarray):models predictions
    """
    # if isinstance(X_test, list):
    #   X_test = np.array(X_test)
    if len(X_test.shape)==1:
      X_test = X_test.reshape(1, -1)
    y_pred = self.model.predict(X_test)
    return y_pred

  def plot_confusion_matrix(self, cm: np.ndarray) -> None:
    """
    This function is used to plot the confusion matrix 
    Argumets:
      cm(np.ndarray):confusion matrix values
    Returns:
    """

    # Normalize confusion matrix (optional)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Create a new figure for the plot
    plt.figure(figsize=(8, 6))

    # Create heatmap with annotations
    sb.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues')
    plt.xlabel('Predicted Price')
    plt.ylabel('True Price')
    plt.title('Confusion Matrix')
    plt.show()
  
  def set_model_name(self, model_name:str='rf'):
    """
    This function is cahnge the models name basically
    Argumets:
      model_name(str):new name for the model
    Returns:
    """
    self.model_name = model_name
