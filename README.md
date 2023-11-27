### EMPLOYEE ATTRITION USING MACHINE LEARNING
Employee attrition prediction using machine learning is a crucial application for organizations aiming to retain talent and reduce turnover. In today's competitive job market, understanding and mitigating employee attrition is paramount. 

Machine learning models, powered by vast data sources and advanced algorithms, offer predictive insights into the factors that contribute to attrition. By analyzing historical data, these models can identify patterns, such as job satisfaction, compensation, and work-life balance, that impact employee turnover. 

This predictive analysis empowers organizations to take proactive measures, such as targeted retention strategies, to minimize attrition and maintain a stable and productive workforce, ultimately enhancing their overall success.

### Hardware Requirements
The hardware requirements for the implementation of the proposed cosmetic product comparison system from handwritten images are outlined below:

### High-Performance Workstation:
A workstation with a multicore processor (e.g., Intel Core i7 or AMD Ryzen 7) for parallel processing.

### Graphics Processing Unit (GPU):
A dedicated GPU (e.g., NVIDIA GeForce RTX series) for accelerated computations, especially for deep learning tasks.

### Memory (RAM):
Minimum 16GB of RAM to handle the computational demands of OCR and image processing tasks.

### Storage:
Adequate storage space (preferably SSD) to accommodate large datasets and model files.

### High-Resolution Display:
A high-resolution 5 for detailed image analysis and visualization.
                                                                                                                                                  
### Software Requirements
The software requirements for the successful deployment of the cosmetic product comparison system are as follows:

### Operating System:
A 64-bit operating system, such as Windows 10 or Ubuntu, for compatibility with modern deep learning frameworks.

### Development Environment:
Python programming language (version 3.6 or later) for coding the OCR
system.

### Machine Learning Frameworks:
Installation of machine learning frameworks, including Logistic regression , to leverage pre-trained models and facilitate model training.

### Flask Libraries:
Integrating dimensionality reduction techniques like t-SNE (t-Distributed Stochastic Neighbor Embedding) from the scikit-learn library within a Flask application can help visualize high-dimensional data effectively.

### Integrated Development Environment (IDE):
Selection of a suitable IDE, such as VSCode or PyCharm, for code development and debugging.

### PROJECT ARCHITECTURE
![WhatsApp Image 2023-11-16 at 9 37 20 PM (1)](https://github.com/rithaniepriyanka/EMPLOYEE/assets/75235132/6256413d-d023-4c81-b27f-17907ccd38ba)

### PROGRAM: 
### DATA PREPROCESSING AND MODEL TRAINING
```
# One HOt Encoding Categorical Features
onehotencoder = OneHotEncoder ()
X_categorical = onehotencoder.fit_transform (X_categorical).toarray ()
X_categorical = pd.DataFrame (X_categorical)
X_categorical
# concat the categorical and numerical values
X_all = pd.concat ([X_categorical,X_numerical],axis=1)
X_all.head ()
# Split Test and Train Data
X_train,X_test,y_train,y_test = train_test_split (X_all,y,test_size=0.20)
# Function that runs the requested algorithm and returns the accuracy metrics
regressor = LogisticRegression ()
regressor.fit (X_train,y_train)
# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
```
### FEATURE TRANSFORMATION AND TARGET LABEL CREATION
```
df['Total_Satisfaction'] = (df['EnvironmentSatisfaction'] +
                                df['JobInvolvement'] +
                                df['JobSatisfaction'] +
                                df['RelationshipSatisfaction'] +
                                df['WorkLifeBalance'])/ 5
    # Drop Columns
    df.drop (   ['EnvironmentSatisfaction','JobInvolvement','JobSatisfaction','RelationshipSatisfaction','WorkLifeBalance'],axis=1,inplace=True)
    # Convert Total satisfaction into boolean
    df['Total_Satisfaction_bool'] = df['Total_Satisfaction'].apply 
                                           (lambda x: 1 if x >= 2.8 else 0)
    df.drop ('Total_Satisfaction',axis=1,inplace=True)
    # It can be observed that the rate of attrition of employees below age 
      of  35 is high
    df['Age_bool'] = df['Age'].apply (lambda x: 1 if x < 35 else 0)
    df.drop ('Age',axis=1,inplace=True)
    # It can be observed that the employees are more likey the drop the job
      if dailyRate less than 800
    df['DailyRate_bool'] = df['DailyRate'].apply (lambda x: 1 if x < 800 
                                                                  else 0)
    df.drop ('DailyRate',axis=1,inplace=True)
    # Employees working at R&D Department have higher attrition rate
    df['Department_bool'] = df['Department'].apply (lambda x: 1 
                           if x == 'Research & Development' else 0)
    df.drop ('Department',axis=1,inplace=True)

    # Rate of attrition of employees is high if DistanceFromHome > 10
    df['DistanceFromHome_bool'] = df['DistanceFromHome'].apply(lambda x: 1
                                                         if x > 10 else 0)
    df.drop ('DistanceFromHome',axis=1,inplace=True)
    # Employees are more likey to drop the job if the employee is working
      as Laboratory Technician
    df['JobRole_bool'] = df['JobRole'].apply (lambda x: 1 
                                 if x == 'Laboratory Technician' else 0)
    df.drop ('JobRole',axis=1,inplace=True)
    # Employees are more likey to the drop the job if the employee's hourly
       rate < 65
    df['HourlyRate_bool'] = df['HourlyRate'].apply (lambda x: 1 
                                                if x < 65 else 0)
    df.drop ('HourlyRate',axis=1,inplace=True)
```
### MAPPING CATEGORICAL FEATURES TO NUMERICAL REPRESENTATIONS
```
       # Convert Categorical to Numerical
       # Business Travel
    if BusinessTravel == 'Rarely':
        df['BusinessTravel_Rarely'] = 1
        df['BusinessTravel_Frequently'] = 0
        df['BusinessTravel_No_Travel'] = 0
    elif BusinessTravel == 'Frequently':
        df['BusinessTravel_Rarely'] = 0
        df['BusinessTravel_Frequently'] = 1
        df['BusinessTravel_No_Travel'] = 0

    else:
        df['BusinessTravel_Rarely'] = 0
        df['BusinessTravel_Frequently'] = 0
        df['BusinessTravel_No_Travel'] = 1
    df.drop ('BusinessTravel',axis=1,inplace=True)
```
### MODEL PREDICTION AND INTERFACE IMPLEMENTATION 
```
prediction = model.predict (df)
    if prediction == 0:
        return render_template ('index.html',prediction_text='Employee Might
                                  Not Leave The Job')
    else:
        return render_template ('index.html',prediction_text='Employee
                                Might Leave The Job')
if _name_ == "_main_":
    app.run (debug=True)
```
### OUTPUT:
![WhatsApp Image 2023-11-16 at 11 45 33 PM](https://github.com/rithaniepriyanka/EMPLOYEE/assets/75235132/411668de-dd70-490b-bf5c-aebad7ef1134)

### RESULT:
Employee Attrition Prediction using Machine Learning holds significant promise for organizations seeking to optimize their workforce management. Through the analysis of historical data and the application of advanced algorithms, this approach empowers businesses to gain valuable insights into employee turnover. Armed with this knowledge, organizations can take proactive measures to retain their talent, reduce turnover costs, and foster a stable and productive workforce.
