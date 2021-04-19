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















