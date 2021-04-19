# Capstone-AzureML
## OverView
![overview](https://user-images.githubusercontent.com/51949018/115182812-3f580a00-a0f8-11eb-9662-c5debd425452.png)

## Dataset
### OverView
According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths.


This dataset is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relavant information about the patient. The dataset is from [Kaggle](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset)

#### Task 
The task, we are going to perform here is find *optimal model* through **HyperDrive** and **AutoML** and deploy that best model for binary classification 

#### Attributes of the Dataset are                                             
| Attributes     | Labels                                             |
|----------------|----------------------------------------------------|
|id              | unique identifier                                  |
|Gender          | Male , Female, other                               |
|Age             | Age of patient                                     |
|HyperTension    | 0 indicates no hypertension,1 indactes hypertension|
|Heart disease   | 0 is for No, 1 is for Yes                          |
|ever_married    | Yes or No                                          |
|Work_type       |Govt_jov",Never_worked, Private or Self-employed    |
|Residence_type  | Rural or Urban                                     |
|avg_glucoselevel|indicates glucose level in body                     |
|bmi             | Body mass index                                    |
|smoking_status  |smokes,unknown,formerly smoked,never smoke          |
|stroke          |if the patient had 1 else 0                         |


### access for the data 
```
found = False
key = 'strokeDataset'
description_text = "Prediction of Stroke"

if key in ws.datasets.keys():
    found = True
    dataset = ws.datasets[key]
    
if not found :
    example = 'https://raw.githubusercontent.com/123manju900/Capstone-AzureML/main/stroke-prediction-dataset.csv'
    dataset = Dataset.Tabular.from_delimited_files(example) 
```
For accessing the Dataset, we can run this command.I have stored the dataset in my Github repo and accessed it 

For registering the model 
```
dataset = dataset.register(workspace = ws,
                          name = key , 
                          description = description_text )
```
Upload of Dataset 
![datasets](https://user-images.githubusercontent.com/51949018/115184858-50a31580-a0fc-11eb-8383-9077df31559d.png)

### list of all Experiments
![all experiments](https://user-images.githubusercontent.com/51949018/115185011-a081dc80-a0fc-11eb-8f33-12ffc2dc293f.png)
`hd-experiment` is the experiment submitted by HyperDrive and `Auto-stoke` is the experiment submitted by AutoML

## AutoML
### AutoML settings
![automlconfig](https://user-images.githubusercontent.com/51949018/115185377-577e5800-a0fd-11eb-8233-dd523106f0a6.png)
`experiment_timeout_minutes`: Here I have given 30 mins of time to run all the algorithms

`max_concurrent_iterations` : Given according to the max nodes allocated to compute

`n_cross_validations`: Number of splits of Data to be split while training the model

`classification` : Here are performing binary classification

`label_column_name` : stroke as we are trying to predict a person has suffered from stroke or not

`enable_early_stopping` : Inorder to avoid unnnecessary usage of compute , it is enabled

`featurization`: Here it is set to auto where it will automatically identify the type of featurization according to the data

## AutoML runwidget
The autoML settings are submitted and Runwidgets is run 
![autoML_widget](https://user-images.githubusercontent.com/51949018/115190110-29047b00-a105-11eb-87f5-cac7d592dcf0.png)

### Completion of AutoML run 
Here we can see the screenshot of AutoMl run
![automate](https://user-images.githubusercontent.com/51949018/115190497-bc3db080-a105-11eb-81b4-fd889d5e328a.png)

## best model
The best model I got is VotingEnsemble
![best algo](https://user-images.githubusercontent.com/51949018/115190961-674e6a00-a106-11eb-8225-20e256e17867.png)

List of other algorithms along with best model 

![best algo 2](https://user-images.githubusercontent.com/51949018/115190982-6cabb480-a106-11eb-94e3-f3abe550dc79.png)

![list of algo](https://user-images.githubusercontent.com/51949018/115191262-d1ffa580-a106-11eb-9af9-72e5010135e5.png)

Parameters of voting ensemble


![metric one](https://user-images.githubusercontent.com/51949018/115191619-4e928400-a107-11eb-904d-48a48c5fd068.png)

![metric 2](https://user-images.githubusercontent.com/51949018/115191646-56eabf00-a107-11eb-8f23-84f1a4823f9f.png)

![metric 3](https://user-images.githubusercontent.com/51949018/115191658-5ce0a000-a107-11eb-97ac-6fb9a4c79f87.png)

![metric 4](https://user-images.githubusercontent.com/51949018/115191671-610cbd80-a107-11eb-9c61-b7f0c325fc75.png)

Other metrics about the best model 
![tag 1](https://user-images.githubusercontent.com/51949018/115192089-f4de8980-a107-11eb-8822-c43c56d0aa52.png)

![tag 2](https://user-images.githubusercontent.com/51949018/115192096-f740e380-a107-11eb-9e2b-ba224da5704d.png)

![fitted model](https://user-images.githubusercontent.com/51949018/115192124-0162e200-a108-11eb-894e-5ea73e25e878.png)


## HyperDrive
For running drive module, I have run HyperDrive  along with **Train.py** file. Since, it is a classification(binary) problem , I have chosen Logistic regression as this runs well with Binary classification 

**Train.py**

In this file I have specified the dataset url which I have stored it on my github and done some featurization. The columns like *Residence_type* , *gender* , *ever_married* , *work_type* were categorical in nature which I have encoded them into mumeric types since logistic regression doesn't support categorical type variables 

### Hyperdriveconfig

**Parameters**

![hyperconfig](https://user-images.githubusercontent.com/51949018/115199321-b7cac500-a110-11eb-89f5-a7f428902d84.png)

`RandomParamtersampling : ` This sampling could could be used for both discrete and continous data 

**parameters I have taken in `RandomParametersampling `**
  
  `C:` This indicates the inverse regularisation strength. Regularization to decrease the cost function. Lesser C value indicates stronger regularization strength
  
  
  `max_iter :` This indicates the number of iterations it is going to perform to get better accuracy on the model
  
  ### Early termination policy 
  
  The policy I have used is *BanditPolicy* 
  
  Bandit policy is based on slack factor/slack amount and evaluation interval. Bandit ends runs when the primary metric isn't within the specified slack factor/slack amount of the most successful run.
  
  slack factor : It defines the slack allowed with respect to the best performing training run.
  
  
  **SKLearn estimator**
  Creates an estimator for training in Scikit-learn experiments.
  
  The **max_total_runs** I have used here is 30 for better model trianing and **max_concurrent_runs** depending upon the maximum nodes allocated
  
  ### RunWidget
  After passing the required parameters in the HyperDriveConfig , I have submitted the run and here are the screenshots of HyperDriveConfig
  
  ![autoMLwidget](https://user-images.githubusercontent.com/51949018/115205593-59edab80-a117-11eb-8d8e-9592a349212b.png)
  
  Widget showing successful runs 
  ![widget2](https://user-images.githubusercontent.com/51949018/115205603-5c500580-a117-11eb-90aa-ff6de98e77ac.png)
  
  Screenshot showing the completed status of HyperDrive experiment
  ![completed status](https://user-images.githubusercontent.com/51949018/115205931-afc25380-a117-11eb-9bc3-8e0cca111964.png)
  
  Graphs related to the runs
  
  ![graph](https://user-images.githubusercontent.com/51949018/115206550-5870b300-a118-11eb-8493-7792d146f0c2.png)
  
  ![childrun](https://user-images.githubusercontent.com/51949018/115206566-5ad30d00-a118-11eb-9ba8-b0bf1482240c.png)
  
  **Best_Run**
  ![bestrun2](https://user-images.githubusercontent.com/51949018/115206728-8524ca80-a118-11eb-9adb-22e253e7ef05.png)
  
  ![best run](https://user-images.githubusercontent.com/51949018/115206740-87872480-a118-11eb-81d0-1c4c991465e4.png)
  Here we can see the accuracy is 0.94 
  
  **best_parameters**
  ![parametric](https://user-images.githubusercontent.com/51949018/115207380-2d3a9380-a119-11eb-95cb-eb57c83e7769.png)
  
  here we can see the best parameters for C is 0.89 and max_iter could be 150 .
  
  
  ## Model deployment 
  Here one may feel that the HyperDriveConfig has performed better than the AutoML but here is something we need to consider as we look at Regularization factor, it is 0.89 which indicates that the cost function is high for this alogorithm. Meaning although it may have given better accuracy on this dataset but it's going to fail on similar data. The maximum number of iterations indicate that the model is over-fitted. So, to come to conclusion here *VotingEnsemble* is the optimal and best algorithm. Voting Ensemble combines more than one algorithm for prediction and predicts using voting count which indiactes it has low-variance to the dataset
  
  ## Deploy model
  
  For deploying a model first we register the best model, for **registering the best model** we can run this code
  ```
  automodel = best_run.register_model(model_name='automl_model', 
                                    model_path='outputs/model.pkl',
                                    tags={'Method':'AutoML'})

print(automodel)
  
  ``` 
  **Once the model is registered, we also need to scoring.py and env.yml files for deployment**
  
  
  *Scoring.py* : This contains all the required configurations for the deployed model 
  *env.yml* : It contains the environment supporting libraries to run the model 
  
  We can download them using the following code
  ```
  # Download scoring file 
best_run.download_file('outputs/scoring_file_v_1_0_0.py', 'score.py')

# Download environment file
best_run.download_file('outputs/conda_env_v_1_0_0.yml', 'env.yml')
d 
```
Now we are going to pass these files to *InferenceConfig* where it is going all the credentials required to deploy the model on the cloud and Finally deploy it using *AciWebservice*

```
inference_config = InferenceConfig(entry_script = script_file, environment = env)

aci_config = AciWebservice.deploy_configuration(cpu_cores = 1,
                                                memory_gb = 1, 
                                                enable_app_insights = True,
                                                auth_enabled = True)
                                            

aci_service_name = 'automl-webservice1'

```

**Deploy the webservice**
![deploy](https://user-images.githubusercontent.com/51949018/115211976-b3f16f80-a11d-11eb-8286-a97445eed0b7.png)

It takes few minutes to deploy the webservice and we can see that the webservice url in the above picture 

**Displaying service Token**


![service token](https://user-images.githubusercontent.com/51949018/115212363-0cc10800-a11e-11eb-83b0-9b0b1f962303.png)


**Dispaly of deployed service in endpoints section**
![proof](https://user-images.githubusercontent.com/51949018/115212719-70e3cc00-a11e-11eb-9e8a-e650d016ee6a.png)

**Webservice showing it is in healthy state**
![healthy](https://user-images.githubusercontent.com/51949018/115212831-8d800400-a11e-11eb-88bc-066569284af3.png)


**While deploying the service, I have enabled `app_insights = True` which gives valuable information regarding the deployed model**
![insight1](https://user-images.githubusercontent.com/51949018/115212862-97096c00-a11e-11eb-94f6-3647d6d6fe2c.png)

![insight2](https://user-images.githubusercontent.com/51949018/115212884-9b358980-a11e-11eb-8574-af9685b522bb.png)

Consuming the *RESTAPI*



![web1-token](https://user-images.githubusercontent.com/51949018/115213719-6d047980-a11f-11eb-9e39-c627a76f9f3b.png)


**TEST**

![test](https://user-images.githubusercontent.com/51949018/115213854-932a1980-a11f-11eb-9d0f-b71531c2d9e5.png)


![test2](https://user-images.githubusercontent.com/51949018/115213866-958c7380-a11f-11eb-8377-3dfcffbe7000.png)

**Service delete**

![servicedelete](https://user-images.githubusercontent.com/51949018/115216511-31b77a00-a122-11eb-928a-9f0704dc7251.png)

## Video
[YouTuBe](https://youtu.be/EXyehG7VO4M)

## Future Improvements
 * Enable ONNX conversion and deploy the model
 * Allow more time to train the data using AutoML for training and check for accuracy 
 * Using SMOTE on the dataset before HyperDrive and check the metrics
 * Train on more data and test the model
 * Deploy the model on IOT azure











































  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  














