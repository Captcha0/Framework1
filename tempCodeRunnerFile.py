model.fit(X_train,Y_train)
# datanew = {
#     'date' : ['2014-07-10 0:00:00'],
#     'bedrooms' : ['3'],
#     'bathrooms' : ['2.5'],
#     'sqft_living' : ['1490'],
#     'sqft_lot' : ['8102'],
#     'floors' : ['2'],
#     'waterfront' : ['0'],
#     'view' : ['0'],
#     'condition' : ['4'],
#     'sqft_above' : ['1490'],
#     'sqft_basement' : ['0'],
#     'yr_built' : ['1990'],
#     'yr_renovated' : ['0'],
# }

# clientesNew = pd.DataFrame(datanew,columns = ['date','bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','sqft_above','sqft_basement','yr_built','yr_renovated'])
# y_pred = model.predict(clientesNew)
# print(clientesNew)
# print(y_pred)