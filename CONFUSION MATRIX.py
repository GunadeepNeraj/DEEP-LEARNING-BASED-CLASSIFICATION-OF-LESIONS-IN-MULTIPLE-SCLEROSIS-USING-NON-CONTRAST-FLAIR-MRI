import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Utility function to get predictions and true labels
def get_preds_and_labels(generator):
    preds = model.predict(generator)
    y_pred = np.argmax(preds, axis=1)
    y_true = generator.classes
    return y_true, y_pred

# ========== CONFUSION MATRIX FOR TRAIN SET ==========
y_true_train, y_pred_train = get_preds_and_labels(train_gen)
cm_train = confusion_matrix(y_true_train, y_pred_train)
print("Train Classification Report:\n", classification_report(y_true_train, y_pred_train))
ConfusionMatrixDisplay(cm_train, display_labels=list(train_gen.class_indices.keys())).plot()
plt.title("Train Set Confusion Matrix")
plt.show()

# ========== CONFUSION MATRIX FOR VAL SET ==========
y_true_val, y_pred_val = get_preds_and_labels(val_gen)
cm_val = confusion_matrix(y_true_val, y_pred_val)
print("Validation Classification Report:\n", classification_report(y_true_val, y_pred_val))
ConfusionMatrixDisplay(cm_val, display_labels=list(val_gen.class_indices.keys())).plot()
plt.title("Validation Set Confusion Matrix")
plt.show()

# ========== CONFUSION MATRIX FOR TEST SET ==========
y_true_test, y_pred_test = get_preds_and_labels(test_gen)
cm_test = confusion_matrix(y_true_test, y_pred_test)
print("Test Classification Report:\n", classification_report(y_true_test, y_pred_test))
ConfusionMatrixDisplay(cm_test, display_labels=list(test_gen.class_indices.keys())).plot()
plt.title("Test Set Confusion Matrix")
plt.show()
