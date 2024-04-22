## Device Price Prediction
This respositery contains the code for the Device prices prediction.

We have 2 main directories:

1. **Python Project**
2. **SpringBoot Project** 

### Run the whole System

Firsly, you need to run the Python project by following the steps:

* clone the project
* open cmd at the python project location
* run **python main.py** you will get the following output
    usage: main.py [-h] --data_path DATA_PATH [--model_name MODEL_NAME] [--imputation_method IMPUTATION_METHOD] [--scaling_data]
               [--normalize] [--pca] [--imputation] [--missing_values_per_feature_display] [--numerical_dist_display]
               [--explore_relash_display] [--corr_display] [--show_evaluation_metrics]
    main.py: error: the following arguments are required: --data_path
to solve this issue you need to specify the path to the dataset for the rest parameters you can keep them as they are or change them their values are mainly boolean and string for the **imputation_method**
* after the project is run you need to run the springboot project

Second, to run the SpringBoot Project:
* clone the project
* open the project and change the credentials for the database at **application.properties**, I have used mysql database and it's local on my machine 
* run the main file

After running both the Python and the SpringBoot projects you can test the api's via postman or any other way to test the endpoints
#### The endpoints in SpringBoot are:

***getAllDevices*** which is a GET api this is the api url (api-url/get_all_devices)
***getDeviceById*** which is a GET api the takes an ID in this form (api-url/{id})
***saveDevice*** which is a POST api the takes an information of a device as JSON body 
***predictPrice*** which is a POST api that takes an ID for a device which then retrieves the information of the device, send it to the Python side and finally returns the prices predicted by the model this is the api url (api-url/predict/{id})

#### The endpoints in Python are:
***predict*** which is a post api that recieves device's features pass it to the model returns the predicted price this is the api url (api-url/predict_price)

In the Python Project folder you will find a file called ***device prices python notebook.ipynb*** which is basically a notebook that you can see some plots and charts and findings about the data and trained models you may also play around with it