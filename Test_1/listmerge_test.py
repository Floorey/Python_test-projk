import json
from faker import Faker
from datetime import date
from collections import Counter
from pprint import pprint
import sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

# Use simple features for clustering 
features = [[len(address)] for address in addresses]

# Standardize the data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Choose the number of clusters
num_clusters = 3

# Perform clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(features_scaled)

# Add cluster information to user data
for i, user in enumerate(user_list):
    user['cluster'] = clusters[i]

# Visualize the clusters
plt.scatter(features_scaled[:, 0], [0] * len(features_scaled), c=clusters, cmap='viridis', alpha=0.5)
plt.title('Address Clustering')
plt.xlabel('Scaled Address Length')
plt.show()

# Assuming you have labels for the clusters in your user list (this is just an example)
labels = [user['cluster'] for user in user_list]

# Assuming you also have other features in your user data
features_for_classification = [[len(user['name']), len(user['email'])] for user in user_list]

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features_for_classification, labels, test_size=0.2, random_state=42)

# Creating and training a classification model
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Making predictions on the test set
predictions = classifier.predict(X_test)

# Evaluating the model accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Model Accuracy: {accuracy}')

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

def serialize_user(obj):
    # Serialize date objects to ISO format
    if isinstance(obj, date):
        return obj.isoformat()
    return obj.__dict__

def main():
    values = loaded_user_list
    analyser = AnalyseUserList(values)
    analyser.analyse_userlist()

if __name__ == "__main__":
    main()
