from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, classification_report, f1_score)
from load_data import load_data
from split_data import sequential_split

x, y = load_data(24)
(xtr, xvalid, xtest, ytr, yvalid, ytest) = sequential_split(x, y)
model = DummyClassifier(strategy="most_frequent")
model.fit(xtr, ytr)
validation_predictions = model.predict(xvalid)
accuracy = accuracy_score(yvalid, validation_predictions)
macro_f1 = f1_score(yvalid, validation_predictions, average="macro")

print("Самый частый класс в train: ")
print(model.classes_[model.class_prior_.argmax()])
print("\nУникальные предсказания модели:")
print(set(validation_predictions))
print(f"\nValidation accuracy: {accuracy:.4f}")
print(f"Validation macro-F1: {macro_f1:.4f}")
print("\nОтчет по классам:")
print(classification_report(yvalid, validation_predictions, digits=3, zero_division=0))