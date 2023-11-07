import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from KNN import KNN


"""
    Author: Mohammad Matin Kateb
    Email: matin.kateb.mk@gmail.com
    GitHub: https://www.github.com/MMatinKateb
"""



def main():
    # Load the digits dataset from scikit-learn
    digits = load_digits()

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

    # Train the KNN model with k=3
    knn = KNN(k=3)
    knn.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = knn.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Plot a sample of the test set with the predicted and actual class labels
    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(8, 3))
    for i in range(10):
        axs[i//5, i%5].imshow(X_test[i].reshape(8, 8), cmap='gray')
        axs[i//5, i%5].set_title("Predicted: {}, Actual: {}".format(y_pred[i], y_test[i]))
        axs[i//5, i%5].axis('off')
    plt.show()




if __name__ == "__main__":
    main()