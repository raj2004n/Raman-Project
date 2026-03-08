import matplotlib.pyplot as plt

def show_results(history, model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    print("Test loss:", loss)
    print("Test accuracy:", accuracy)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['training', 'validation'], loc='best')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training', 'validation'], loc='best')
    plt.show()