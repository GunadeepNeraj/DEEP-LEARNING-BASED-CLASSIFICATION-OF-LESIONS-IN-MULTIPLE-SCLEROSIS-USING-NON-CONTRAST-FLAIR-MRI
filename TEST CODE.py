import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from tensorflow.keras.preprocessing import image

# ======= PARAMETERS =======
model_path = 'h12.h5'
test_folder = '/content/drive/MyDrive/test7'  # Assumes subfolders like /test7/ms and /test7/noms
img_size = (160, 160)
batch_size = 16
class_labels = ['ms', 'noms']  # Adjust if needed

# ======= LOAD MODEL =======
model = load_model(model_path)
print("✅ Model loaded.")

# ======= PREPARE TEST DATA =======
test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    test_folder,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# ======= PREDICT AND EVALUATE =======
y_true = test_gen.classes
y_pred_probs = model.predict(test_gen)
y_pred = np.argmax(y_pred_probs, axis=1)

# Accuracy
acc = accuracy_score(y_true, y_pred)
print(f"✅ Test Accuracy: {acc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(cmap='Blues')
plt.title("Test Set Confusion Matrix")
plt.show()

# ======= VISUALIZE SAMPLE PREDICTIONS =======
filenames = test_gen.filenames
plt.figure(figsize=(14, 8))
for i in range(8):
    img_path = os.path.join(test_folder, filenames[i])
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    label = class_labels[np.argmax(pred)]

    plt.subplot(2, 4, i+1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Pred: {label}")

plt.tight_layout()
plt.show()
