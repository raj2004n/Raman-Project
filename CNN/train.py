from model import *
from data import *
from evaluate import *
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping
import pickle
from collections import Counter

spectra_list, metadata = get_data()

y_data = [m['##NAMES'].split(',')[0].strip() for m in metadata]

train_index, test_index = leave_one_out_split(y_data)

#TODO: First make the model match the paper more
#TODO: The consider augmenting more data
# one hot encoding 
le = LabelEncoder()
le.fit(y_data)

classes = le.classes_
num_classes = len(classes)

y_data = np.array(y_data)

y_train_labels = y_data[train_index]
y_test_labels = y_data[test_index]

y_train = to_categorical(le.transform(y_train_labels), num_classes)
y_test = to_categorical(le.transform(y_test_labels), num_classes)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

spectra_train = [spectra_list[i] for i in train_index]
spectra_test = [spectra_list[i] for i in test_index]

# hard coded for now
x_min, x_max = 100, 1800

x_train = standardise_data(spectra_train, target_length=913, x_min=x_min, x_max=x_max)
x_test = standardise_data(spectra_test, target_length=913, x_min=x_min, x_max=x_max)

# only apply augmentation to training set
x_train_aug, y_train_aug_labels = build_augmented_dataset(x_train, y_train_labels, n_shift=1, n_noise=1, n_combinations=1)

y_train_aug = to_categorical(le.transform(y_train_aug_labels), num_classes)
y_train_aug_integers = np.argmax(y_train_aug, axis=1)

print(f"Original training samples: {len(x_train)}")
print(f"Augmented training samples: {len(x_train_aug)}")

classes_array = np.arange(num_classes)
class_weights = compute_class_weight(class_weight='balanced', classes=classes_array, y=y_train_aug_integers)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights - min: {min(class_weights):.3f}, max: {max(class_weights):.3f}")

aug_counts = Counter(y_train_aug_labels)

total_original  = len(x_train)
total_augmented = len(x_train_aug)
total_generated = total_augmented - total_original

print(f"\n--- Augmentation Summary ---")
print(f"Original training samples:  {total_original}")
print(f"Total after augmentation:   {total_augmented}")
print(f"Generated samples:          {total_generated}")
print(f"Augmentation multiplier:    {total_augmented / total_original:.1f}x")

model = CNN_model(num_classes)
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True)


early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    x_train_aug, y_train_aug, epochs=100,
    batch_size=64,
    validation_data=(x_test, y_test),
    callbacks=[early_stop],
    class_weight=class_weight_dict
    )

test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

# save full model and wavenumber range
model.save("raman_cnn_model.keras")
model.save_weights("test_weights.weights.h5")
np.save("wavenumber_range.npy", np.array([x_min, x_max]))

show_results(history, model, x_test, y_test)