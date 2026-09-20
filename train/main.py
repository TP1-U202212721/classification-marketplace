import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

dataset = pd.read_csv('train/dataset/dataset_fb_marketplace.csv')
labels = dataset['etiqueta_riesgo']
features = dataset.drop(columns=['etiqueta_riesgo'])

X_train, X_test, y_train, y_test = train_test_split(
    features, 
    labels, 
    test_size=0.2, 
    random_state=42
)


model = RandomForestClassifier(
    max_depth=10,
    criterion='log_loss',
    n_estimators=100, 
    random_state=42,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

joblib.dump(model, 'train/models/model_fb_marketplace.pkl')



