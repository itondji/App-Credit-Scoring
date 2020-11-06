from flask import Flask,render_template,url_for,request,session,redirect,url_for
from flask_material import Material

# import EDA Packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import ML Packages

# joblib or pickle are for loading the model (already built)

#from sklearn.externals import joblib 

import joblib

app=Flask(__name__)
#Material(app)

@app.route('/')

def index():
	return render_template("index.html")

@app.route('/news')
def news():
	return "<h1> Hello </h1>"

@app.route('/analyze',methods=['POST'])

def analyze():
	if request.method=='POST':
		ApplicantIncome=request.form['ApplicantIncome']
		CoapplicantIncome=request.form['CoapplicantIncome']
		LoanAmount=request.form['LoanAmount']
		Loan_Amount_Term=request.form['Loan_Amount_Term']

		Credit_History=request.form['Credit_History']
		Gender=request.form['Gender']
		Married=request.form['Married']
		Education=request.form['Education']
		Dependents=request.form['Dependents']
		Self_Employed=request.form['Self_Employed']
		Property_Area=request.form['Property_Area']
		model_choice=request.form['model_choice']
		
		## Variables name: data
		sample_data=[Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area]
		ex=np.array(sample_data).reshape(-1,1)

		ex=pd.DataFrame(ex).T
		ex.columns=['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area']

		mode=['Male', 'Yes', '0', 'Graduate', 'No', 'Semiurban']

		Cat_variables=['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
		Numerical_variables=['ApplicantIncome','CoapplicantIncome','LoanAmount','Credit_History']

		#Missing values: categorical
		#for i in range(len(Cat_variables)):
			#l=Cat_variables[i]
    		#ex[[l]]=ex[[l]].fillna(mode[i])

    
		Property_Area_code={'Semiurban':2,'Urban':3,"Rural":0}
		Dependents_code={'0':0,'1':1,'2':2,'3+':3}
		Gender_code={'Male':1,'Female':0}
		Married_code={'Yes':1,'No':0}
		Self_Employed_code={'No':1,'Yes':0}
		Education_code={'Graduate':2,'Not Graduate':1}

    	## Missing values: num
		#df.fillna(0,axis=0,inplace=True)

		ex.Education=ex.Education.map(Education_code)
		ex.Property_Area=ex.Property_Area.map(Property_Area_code)
		ex.Married=ex.Married.map(Married_code)
		ex.Dependents=ex.Dependents.map(Dependents_code)
		ex.Gender=ex.Gender.map(Gender_code)
		ex.Self_Employed=ex.Self_Employed.map(Self_Employed_code)

		## variance and mean

		var_num=[3.61627278e+07, 7.25214502e+06, 6.66454451e+03, 1.80437280e-01]
		mean=[5.31788187e+03, 1.57956196e+03, 1.38775967e+02, 7.63747454e-01]

		ex_num=ex[Numerical_variables]
		ex=ex.drop(Numerical_variables,axis=1)

		## Standardization
		ex_num=ex_num.astype(float)
		ex_cale=(ex_num-mean)/var_num
		ex_cale=pd.DataFrame(ex_cale,columns=Numerical_variables)
		new_data=pd.concat([ex_cale,ex],axis=1)

		# ML conditional: 
		if model_choice=='logitmodel':
			logit_model=joblib.load('./data/logit_model_credit.pkl')
			result_prediction=logit_model.predict(new_data)
		elif model_choice=='knnmodel':
			knn_model=joblib.load('./data/knn_model_credit.pkl')
			result_prediction=knn_model.predict(new_data)
		elif model_choice=='svmmodel':
			svm_model=joblib.load('./data/svm_model_credit.pkl')
			result_prediction=svm_model.predict(new_data)
		else:
			logit_model=joblib.load('./data/logit_model_credit.pkl')
			result_prediction=logit_model.predict(new_data)

	return render_template("index.html",ApplicantIncome=ApplicantIncome, CoapplicantIncome=CoapplicantIncome,
		LoanAmount=LoanAmount,Credit_History=Credit_History,Gender=Gender,Married=Married,Education=Education,Dependents=Dependents,Self_Employed=Self_Employed,Property_Area=Property_Area,
		clean_data=ex,result_prediction=result_prediction)

if __name__ == '__main__':
	app.run(debug=True)