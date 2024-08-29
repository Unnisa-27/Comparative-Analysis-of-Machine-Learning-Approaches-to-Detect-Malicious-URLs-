models=pd.DataFrame({
    'Model': ['Random Forest', 'CATB', 'Naive bayes'],
    'Test Score': [rf_acc,cat_acc,nb_acc]})
models.sort_values(by='Test Score',ascending=False)
# Creating holders to store the model performance results
models = ['Random Forest', 'CATB', 'Naive bayes']
train_accuracies = [rf_model_rfe.score(X_train_rf_rfe, y_train), cat_model_rfe.score(X_train_cat_rfe, y_train), categorical_nb_model.score(X_train_nb, y_train)]
test_accuracies = [rf_acc, cat_acc, nb_acc]

#creating dataframe
results = pd.DataFrame({ 'ML Model': models,
    'Train Accuracy': train_accuracies,
    'Test Accuracy': test_accuracies})
results
#Sorting the datafram on accuracy
results.sort_values(by=['Test Accuracy', 'Train Accuracy'], ascending=False)
from sklearn.metrics import classification_report
#Evaluate the models
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_y_pred_rfe))

print("\nCatBoost Classification Report:")
print(classification_report(y_test, cat_y_pred_rfe))

print("\nNaive Bayes Classification Report:")
print(classification_report(y_test, categorical_nb_y_pred))

from sklearn.metrics import confusion_matrix
# we just print the confusion matrix for each model
#confusion matrix, accuracy, recall, precision, f1_score
# Calculate the metrics for each model
model_metrics = {}
predictions=[rf_y_pred_rfe,cat_y_pred_rfe,categorical_nb_y_pred]
for model,prediction in zip(models,predictions):
    con_matrix=confusion_matrix(y_test,prediction)

    tn,fp,fn,tp=con_matrix.ravel()

    accuracy=(tp+tn)/(tp+tn+fp+fn)

    precision=tp/(tp+fp)

    recall=tp/(tp+fn)

    f1_score=2 * precision *recall/ (precision+recall)

    model_metrics[model] = {
        'Accuracy': accuracy,
        'Recall': recall,
        'Precision': precision,
        'F1 Score': f1_score
    }

    print('Model: {}'.format(model))
    print('Confusion Matrix')
    print(con_matrix)

    print('Accuracy: {:.3f}'.format(accuracy))
    print('Recall: {:.3f}'.format(recall))
    print('Precision: {:.3f}'.format(precision))
    print('F1 score: {:.3f}'.format(f1_score))
    print('\n')

# Find the best performer based on the F1 Score
best_model = max(model_metrics, key=lambda x: model_metrics[x]['F1 Score'])
print(f"Best Model: {best_model} with F1 Score: {model_metrics[best_model]['F1 Score']:.3f}")
