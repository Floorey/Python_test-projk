import json
from faker import Faker
from datetime import date
from collections import Counter
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim, GoogleV3

fake = Faker()

def serialize_user(obj):
    # Serialize date objects to ISO format
    if isinstance(obj, date):
        return obj.isoformat()
    return obj.__dict__

def generate_random_user():
    # Generate a random user with fake data
    user = {
        'name': fake.name(),
        'address': fake.address(),
        'email': fake.email(),
        'phone_number': fake.phone_number(),
        'date_of_birth': fake.date_of_birth(),
    }
    return user

user_list = []

# Generate 100 random users and save to JSON file
for _ in range(100):
    random_user = generate_random_user()
    user_list.append(random_user)

with open('user_data.json', 'w') as json_file:
    json.dump(user_list, json_file, default=serialize_user, indent=2)

with open('user_data.json', 'r') as json_file:
    loaded_user_list = json.load(json_file)

# Work with addresses for clustering
addresses = [user['address'] for user in user_list]

# Extract zip codes from addresses using Nominatim
geolocator_nominatim = Nominatim(user_agent="address_clustering")
zip_codes_nominatim = []

for address in addresses:
    try:
        location = geolocator_nominatim.geocode(address)
        if location and location.address and 'postcode' in location.raw['address']:
            zip_code = location.raw['address']['postcode']
            zip_codes_nominatim.append(zip_code)
    except GeocoderTimedOut:
        pass

# Extract zip codes from addresses using GoogleV3
geolocator_google = GoogleV3(api_key='AIzaSyChyjjJIthtCB1BSJXc2VBteoIZ0zpqsEI')  
zip_codes_google = []

for address in addresses:
    try:
        location = geolocator_google.geocode(address)
        if location and location.address and 'postal_code' in location.raw['address_components']:
            zip_code = location.raw['address_components']['postal_code']
            zip_codes_google.append(zip_code)
    except GeocoderTimedOut:
        pass

# Convert zip codes to numerical values
numeric_zip_codes_nominatim = [int(zip_code) for zip_code in zip_codes_nominatim]
numeric_zip_codes_google = [int(zip_code) for zip_code in zip_codes_google]

# Use zip codes as features for clustering (using Nominatim)
features_nominatim = [[zip_code] for zip_code in numeric_zip_codes_nominatim]

# Standardize the data
scaler_nominatim = StandardScaler()
features_scaled_nominatim = scaler_nominatim.fit_transform(features_nominatim)

# Perform clustering (using Nominatim)
num_clusters_nominatim = 3
kmeans_nominatim = KMeans(n_clusters=num_clusters_nominatim, random_state=42)
clusters_nominatim = kmeans_nominatim.fit_predict(features_scaled_nominatim)

# Add cluster information to user data
for i, user in enumerate(user_list):
    user['cluster_nominatim'] = clusters_nominatim[i]

# Visualize the clusters (using Nominatim)
plt.scatter(features_scaled_nominatim[:, 0], [0] * len(features_scaled_nominatim), c=clusters_nominatim, cmap='viridis', alpha=0.5)
plt.title('Address Clustering based on Zip Codes (Nominatim)')
plt.xlabel('Numerical Zip Codes')
plt.show()

# Use zip codes as features for clustering (using GoogleV3)
features_google = [[zip_code] for zip_code in numeric_zip_codes_google]

# Standardize the data
scaler_google = StandardScaler()
features_scaled_google = scaler_google.fit_transform(features_google)

# Perform clustering (using GoogleV3)
num_clusters_google = 3
kmeans_google = KMeans(n_clusters=num_clusters_google, random_state=42)
clusters_google = kmeans_google.fit_predict(features_scaled_google)

# Add cluster information to user data
for i, user in enumerate(user_list):
    user['cluster_google'] = clusters_google[i]

# Visualize the clusters (using GoogleV3)
plt.scatter(features_scaled_google[:, 0], [0] * len(features_scaled_google), c=clusters_google, cmap='viridis', alpha=0.5)
plt.title('Address Clustering based on Zip Codes (GoogleV3)')
plt.xlabel('Numerical Zip Codes')
plt.show()

# Assuming you have labels for the clusters in your user list (this is just an example)
labels_nominatim = [user['cluster_nominatim'] for user in user_list]
labels_google = [user['cluster_google'] for user in user_list]

# Assuming you also have other features in your user data
features_for_classification_nominatim = [[len(user['name']), len(user['email'])] for user in user_list]
features_for_classification_google = [[len(user['name']), len(user['email'])] for user in user_list]

# Splitting the data into training and test sets (using Nominatim)
X_train_nominatim, X_test_nominatim, y_train_nominatim, y_test_nominatim = train_test_split(features_for_classification_nominatim, labels_nominatim, test_size=0.2, random_state=42)

# Creating and training a classification model (using Nominatim)
classifier_nominatim = RandomForestClassifier(random_state=42)
classifier_nominatim.fit(X_train_nominatim, y_train_nominatim)

# Making predictions on the test set (using Nominatim)
predictions_nominatim = classifier_nominatim.predict(X_test_nominatim)

# Evaluating the model accuracy (using Nominatim)
accuracy_nominatim = accuracy_score(y_test_nominatim, predictions_nominatim)
print(f'Model Accuracy (Nominatim): {accuracy_nominatim}')

# Splitting the data into training and test sets (using GoogleV3)
X_train_google, X_test_google, y_train_google, y_test_google = train_test_split(features_for_classification_google, labels_google, test_size=0.2, random_state=42)

# Creating and training a classification model (using GoogleV3)
classifier_google = RandomForestClassifier(random_state=42)
classifier_google.fit(X_train_google, y_train_google)

# Making predictions on the test set (using GoogleV3)
predictions_google = classifier_google.predict(X_test_google)

# Evaluating the model accuracy (using GoogleV3)
accuracy_google = accuracy_score(y_test_google, predictions_google)
print(f'Model Accuracy (GoogleV3): {accuracy_google}')

class AnalyseUserList:
    def __init__(self, values):
        self.values = values

    def analyse_userlist(self):
        # Extract addresses from user data
        addresses = [user['address'] for user in self.values]
        # Count occurrences of each address
        address_counts = Counter(addresses)
        # Find the most common address and its count
        most_common_address, count = address_counts.most_common(1)[0]
        print(f'The most common address is: {most_common_address} (occurs {count} times)')

def main():
    values = loaded_user_list
    analyser = AnalyseUserList(values)
    analyser.analyse_userlist()

if __name__ == "__main__":
    main()
