import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import IsolationForest

class MachineLearningMain():
    def __init__(self):
        self.model = LinearRegression()
        self.outlier_detector = IsolationForest()

    def detect_outliers(self, X):
        # Ausreißer erkennen
        outliers = self.outlier_detector.fit_predict(X)
        return outliers

    def train_from_file(self, file_path):
        # Einlesen der CSV-Datei mit Pandas
        df = pd.read_csv(file_path)
        
        if df.shape[0] >= 5:
            # Überspringen von Kommentarzeilen, die mit 'values' beginnen
            df = df[~df.iloc[:, 0].str.startswith('values')]
            
            X = df.iloc[:, 2:].values  # Eingangsdaten: alle Spalten außer der ersten
            y = df.iloc[:, 2].values    # Zielvariable: erste Spalte

            # Ausreißer erkennen
            outliers = self.detect_outliers(X)
            
            # Nur nicht-ausreißerische Daten verwenden
            X_filtered = X[outliers == 1]
            y_filtered = y[outliers == 1]

            self.model.fit(X_filtered, y_filtered)
            y_pred = self.model.predict(X_filtered)
            
            mse = mean_squared_error(y_filtered, y_pred)
            print(f'Mean Squared Error (after removing outliers): {mse}')
        else:
            print('Nicht genügend Spalten in den Daten vorhanden.')

        return self.model


# Beispiel für die Verwendung der ML-Klasse
if __name__ == "__main__":
    ml_instance = MachineLearningMain()
    file_path = r"C:\Users\lukas\OneDrive\Dokumente\Python_test-projk\exaplle.csv"

    trained_model = ml_instance.train_from_file(file_path)
