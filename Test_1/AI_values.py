import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import IsolationForest

class MachineLearningMain():
    def __init__(self):
        self.model = LinearRegression()
        self.outlier_detector = IsolationForest()
        self.lower_bound = 0
        self.upper_bound = 1

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
            
            X = df.iloc[:, 2:].values  # Eingangsdaten: alle Spalten außer der ersten 2
            y = df.iloc[:, 2].values    # Zielvariable: ersten 2 Spalte

            # Ausreißer erkennen
            outliers = self.detect_outliers(X)
            
            # Nur nicht-ausreißerische Daten verwenden
            X_filtered = X[outliers == 1]
            y_filtered = y[outliers == 1]

            self.model.fit(X_filtered, y_filtered)
            self.y_pred = self.model.predict(X_filtered)

             # Grenzwerte für den erwarteten Bereich anpassen
            self.lower_bound = min(self.y_pred)
            self.upper_bound = min(self.y_pred)

            print(f'Lower Bound: {self.lower_bound}, Upper Bound: {self.upper_bound}')  # Ausgabe der Grenzwerte
            
           
            mse = mean_squared_error(y_filtered, self.y_pred)
            print(f'Mean Squared Error (after removing outliers): {mse}')
        else:
            print('Nicht genügend Spalten in den Daten vorhanden.')

        return self.model
    
    def predict_user_input(self):
        user_input = input('Geben Sie einen Wert ein: ')


        if not user_input.strip():
            print('No input!')
            return
        
        
        try:

            user_value = float(user_input.strip())  # Versuchen, den Benutzereingabewert in eine Gleitkommazahl umzuwandeln
        except ValueError:
            print('Ungültige Eingabe. Bitte geben Sie eine gültige Zahl ein.')
            return

        prediction = self.model.predict([[user_value]])  # Vorhersage machen

       


        if prediction >= self.lower_bound and prediction <= self.upper_bound:
            print('Der eingegebene Wert liegt im erwarteten Bereich!')
        else:
            print('Der eingegebene Wert liegt außerhalb des erwarteten Bereichs!')


# Beispiel für die Verwendung der ML-Klasse
if __name__ == "__main__":
    ml_instance = MachineLearningMain()
    base_file_path = r'example'
    num_files = 50

    for i in range(num_files):
        file_path = f'{base_file_path}_{i+1}.csv'
        trained_model = ml_instance.train_from_file(file_path)

    ml_instance.predict_user_input()  # Benutzereingabe vorhersagen
