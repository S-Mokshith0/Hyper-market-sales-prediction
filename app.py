from flask import Flask, render_template
from flask import request
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('Train.csv')
df.describe()
df.info()
print(df.apply(lambda x: len(x.unique())))

print(df.isnull().sum())
print(df.head())
cat_col = []
for x in df.dtypes.index:
    if df.dtypes[x] == 'object':
        cat_col.append(x)

print(cat_col)
cat_col.remove('Item_Identifier')
cat_col.remove('Outlet_Identifier')
for col in cat_col:
    print(col)
    print(df[col].value_counts())
    print()

item_weight_mean = df.pivot_table(values = "Item_Weight", index = 'Item_Identifier')
miss_bool = df['Item_Weight'].isnull()
for i, item in enumerate(df['Item_Identifier']):
    if miss_bool[i]:
        if item in item_weight_mean:
            df['Item_Weight'][i] = item_weight_mean.loc[item]['Item_Weight']
        else:
            df['Item_Weight'][i] = np.mean(df['Item_Weight'])

df['Item_Weight'].isnull().sum()

outlet_size_mode = df.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))

miss_bool = df['Outlet_Size'].isnull()

df.loc[miss_bool, 'Outlet_Size'] = df.loc[miss_bool, 'Outlet_Type'].apply(lambda x: outlet_size_mode[x])

df.loc[:, 'Item_Visibility'].replace([0], [df['Item_Visibility'].mean()], inplace=True)

df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF':'Low Fat', 'reg':'Regular', 'low fat':'Low Fat'})

df['New_Item_Type'] = df['Item_Identifier'].apply(lambda x: x[:2])

df['New_Item_Type'] = df['New_Item_Type'].map({'FD':'Food', 'NC':'Non-Consumable', 'DR':'Drinks'})

df.loc[df['New_Item_Type']=='Non-Consumable', 'Item_Fat_Content'] = 'Non-Edible'

df['Outlet_Years'] = 2013 - df['Outlet_Establishment_Year']

print("fafasdfsadf")
print(df.head())

corr = df.corr()
print(corr)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Outlet'] = le.fit_transform(df['Outlet_Identifier'])
cat_col = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'New_Item_Type']
for col in cat_col:
    df[col] = le.fit_transform(df[col])
print(df.head())

df = pd.get_dummies(df, columns=['Item_Fat_Content', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'New_Item_Type'])
df.head()

X = df.drop(columns=['Outlet_Establishment_Year', 'Item_Identifier', 'Outlet_Identifier','Item_MRP'])
y = df['Item_Outlet_Sales']
X=df[['Item_Weight','Item_Visibility','Item_Type','Outlet_Years']]
X.head(1)
x1={'Item_Weight':[20.75],'Item_Visibility':[0.007564836],'Item_Type':[2],'Outlet_Years':[14]}
X1=pd.DataFrame(x1)

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error


def train(model, X, y):
    # train the model
    model.fit(X, y)
    # predict the training set
    pred = model.predict(X)

    # perform cross-validation
    cv_score = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
    cv_score = np.abs(np.mean(cv_score))
    print(pred)
    print("Model Report")
    print("MSE:", mean_squared_error(y, pred))
    print("CV Score:", cv_score)
    print("\n")
    return pred

def train1(model, X, y,X1):
    # train the model
    model.fit(X, y)
    # predict the training set
    pred = model.predict(X)
    pred1=model.predict(X1)
    # perform cross-validation
    cv_score = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
    cv_score = np.abs(np.mean(cv_score))
    print("Model Report")
    print("MSE:", mean_squared_error(y, pred))
    print("CV Score:", cv_score)
    print(pred1)
    print("\n")
    return pred1

def train2(model, y,X1):
    # train the model
    model.fit(X, y)
    # predict the training set
    pred1=model.predict(X1)
    # perform cross-validation
    
    print(pred1)
    print("\n")
    return pred1

from sklearn.linear_model import LinearRegression, Ridge, Lasso
model = LinearRegression(normalize=True)
train(model, X, y)
coef = pd.Series(model.coef_, X.columns).sort_values()
coef.plot(kind='bar', title="Model Coefficients")

model = Ridge(normalize=True)
train(model, X, y)

model = Lasso()
train(model, X, y)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
train(model, X, y)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
train(model, X, y)

from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor()
train(model, X, y)

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/predict.html')
def predict():
    return render_template('predict.html')

@app.route('/ans.html',methods = ['POST'])
def ans():
    uname=request.form['uname']  
    passwrd=request.form['pass'] 
    uname1=request.form['uname1']  
    passwrd1=request.form['pass1']  
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor()
    x1={'Item_Weight':[uname],'Item_Visibility':[passwrd],'Item_Type':[uname1],'Outlet_Years':[passwrd1]}
    X1=pd.DataFrame(x1)

    y=df['Item_MRP']
    item_mrp=train1(model, X, y,X1)
    smalllist= str(item_mrp.tolist())
    model = DecisionTreeRegressor()
    print(X1)
    y = df['Item_Outlet_Sales']
    item_outlet_sales=train2(model, y,X1)
    ios= str(item_outlet_sales.tolist())
    model = DecisionTreeRegressor()
    y=df['Outlet_Size_0']
    outlet_size_0=train1(model, X, y,X1)
    os0= str(outlet_size_0.tolist())
    os='Small'
    if outlet_size_0[0]==0:
        y=df['Outlet_Size_1']
        outlet_size_1=train1(model, X, y,X1)
        os1= str(outlet_size_1.tolist())
        os='Medium'
        if outlet_size_1[0]==0:
            y=df['Outlet_Size_2']
            outlet_size_1=train1(model, X, y,X1)
            os1= str(outlet_size_1.tolist())
            os='High'
    return render_template("ans.html",itemmrp=smalllist[1:len(smalllist)-1],itemos=ios[1:len(ios)-1],outletsize=os,u=uname,v=passwrd,w=uname1,y=passwrd1)

@app.route('/allans.html',methods = ['POST'])
def allans():
    uname=request.form['u1']  
    passwrd=request.form['v1'] 
    uname1=request.form['w1']  
    passwrd1=request.form['y1']  
    x1={'Item_Weight':[uname],'Item_Visibility':[passwrd],'Item_Type':[uname1],'Outlet_Years':[passwrd1]}
    X1=pd.DataFrame(x1)
    y = df['Item_Outlet_Sales']
    model = DecisionTreeRegressor()
    item_outlet_sales=train1(model, X, y,X1)
    d= str(item_outlet_sales.tolist())
    model = LinearRegression(normalize=True)
    item_outlet_sales=train1(model, X, y,X1)
    lr= str(item_outlet_sales.tolist())
    model = Ridge(normalize=True)
    item_outlet_sales=train1(model, X, y,X1)
    r= str(item_outlet_sales.tolist())
    model = Lasso()
    item_outlet_sales=train1(model, X, y,X1)
    l= str(item_outlet_sales.tolist())
    model = RandomForestRegressor()
    item_outlet_sales=train1(model, X, y,X1)
    rfr= str(item_outlet_sales.tolist())
    model = ExtraTreesRegressor()
    item_outlet_sales=train1(model, X, y,X1)
    etr= str(item_outlet_sales.tolist())
    return render_template("allans.html",d1=d[1:len(d)-1],lr1=lr[2:len(lr)-1],r1=r[2:len(r)-1],l1=l[2:len(l)-1],rfr1=rfr[2:len(rfr)-1],etr1=etr[2:len(etr)-1])
if __name__== "__main__":
    app.run(debug=True)