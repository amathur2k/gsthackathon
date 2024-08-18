import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def impute_df_preserve_id(df, id_col='ID', strategy='mean'):
    dummy = df.drop(id_col, axis=1)
    imputed = pd.DataFrame(SimpleImputer(strategy=strategy).fit_transform(dummy), columns=dummy.columns)
    return pd.concat([df[id_col], imputed], axis=1)

# Load datasets
x = pd.read_csv('X_Train_Data_Input.csv', encoding='ascii', nrows = 800000)
y = pd.read_csv('Y_Train_Data_Target.csv', encoding='ascii', nrows = 800000)

# Merge datasets
full_df = pd.merge(x, y, on='ID')

# Separate features and target
#X_full = full_df.drop(['ID', 'target'], axis=1)
#y_full = full_df['target']

# Impute missing values
full_df_imputed = impute_df_preserve_id(full_df)
X_full_imputed = full_df_imputed.drop(['ID', 'target'], axis=1)
Y_full_imputed = full_df_imputed['target']

# Split the full_df dataframe into training and testing datasets
train_imputed, test_imputed = train_test_split(full_df_imputed, test_size=0.2, random_state=42)
X_train_imputed = train_imputed.drop(['ID', 'target'], axis=1)
Y_train_imputed = train_imputed['target']

# Define class weights
class_weights = {0: 1.0, 1: 5}

# Train Random Forest model
rf_full_model = RandomForestClassifier(class_weight=class_weights, random_state=42, n_jobs=-1)
#rf_full_model.fit(X_full_imputed, y_full)
rf_full_model.fit(X_train_imputed, Y_train_imputed)
print ("Done with the RF")

# Train Naive Bayes model
nb_model = GaussianNB(priors=[0.1,0.9])
nb_model.fit(X_train_imputed, Y_train_imputed)
print ("Done with the NB")

# Train XGBoost model
#xgb_full_model = xgb.XGBClassifier(scale_pos_weight=5, random_state=42, use_label_encoder=False, eval_metric='logloss')
#xgb_full_model.fit(X_train_imputed, Y_train_imputed)
#print ("Done with the xgboost")

# Train Gradient Boosting model
gb_full_model = GradientBoostingClassifier(random_state=42)
sample_weight = np.where(Y_train_imputed == 1, 5, 1)
gb_full_model.fit(X_train_imputed, Y_train_imputed, sample_weight=sample_weight)
print ("Done with the GB")

#Train a SVm Model
#svm_model = SVC(kernel='rbf', probability=False, gamma='scale', class_weight=class_weights)
#svm_model.fit(X_train_imputed, Y_train_imputed)

#print ("Done with the rbf kernel")
#Train a SVM Poly model
#svm_model_poly = SVC(kernel='poly', degree=3, probability=True, gamma='scale', class_weight=class_weights)
#svm_model_poly.fit(X_train_imputed, Y_train_imputed)

# Predict scores using the trained models
rf_full_scores = rf_full_model.predict_proba(X_full_imputed)[:, 1]
nb_scores = nb_model.predict_proba(X_full_imputed)[:, 1]

#xgb_full_scores = xgb_full_model.predict_proba(X_full_imputed)[:, 1]
gb_full_scores = gb_full_model.predict_proba(X_full_imputed)[:, 1]
#svm_scores = svm_model.predict_proba(X_full_imputed)[:, 1]

#svm_scores = svm_model.decision_function(X_full_imputed)

#svm_scores_poly = svm_model_poly.predict_proba(X_full_imputed)[:, 1]


# Create a new dataframe with ID and prediction scores
predictions_df = pd.DataFrame({
    'ID': full_df['ID']
    ,'RF_Score': rf_full_scores
    ,'NB_Score': nb_scores
    #,'XGB_Score': xgb_full_scores,
    ,'GB_Score': gb_full_scores
    #,'SVM_Score': svm_scores
    #,'SVM_Score_Poly': svm_scores_poly

})

# Merge predictions with the original merged dataset
final_df = pd.merge(full_df_imputed, predictions_df, on='ID')

#product = final_df['Column1'] * final_df['Column2']
#final_df['Column12rms'] = np.where(product >= 0, np.sqrt(product), 0)

#final_df['ColumnRFMod'] =  np.square(final_df['RF_Score'] * 10)
#final_df['ColumnXGBMod'] =  np.square(final_df['XGB_Score'] * 10)
#final_df['ColumnGBMod'] =  np.square(final_df['GB_Score'] * 10)





# Save the final dataframe to a CSV file
final_df.to_csv('training_data.csv', index=False)
print("Final dataframe saved to 'training_data.csv'")

# Display some information about the final dataset
print("\nShape of final dataframe:", final_df.shape)
print("\nColumns in the final dataframe:", final_df.columns.tolist())
print("\nFirst few rows of the final dataframe:")
print(final_df.head())