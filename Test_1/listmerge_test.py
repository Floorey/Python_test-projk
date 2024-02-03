import json
from faker import Faker
from datetime import date
from collections import Counter
from pprint import pprint


fake = Faker()


class AnalyseUserList:
    def __init__(self, values):
        self.values = values

    def analyse_userlist(self):
        addresses = [user['address'] for user in self.values]
        address_counts = Counter(addresses)
        most_common_address, count = address_counts.most_common(1)[0]
        print(f'The most common address is: {most_common_address} (occurs {count} times)')


def serialize_user(obj):
    if isinstance(obj, date):
        return obj.isoformat()
    return obj.__dict__


def generate_random_user():
    user = {
        'name': fake.name(),
        'address': fake.address(),
        'email': fake.email(),
        'phone_number': fake.phone_number(),
        'date_of_birth': fake.date_of_birth(),
    }
    return user


user_list = []

for _ in range(100):
    random_user = generate_random_user()
    user_list.append(random_user)

with open('user_data.json', 'w') as json_file:
    json.dump(user_list, json_file, default=serialize_user, indent=2)

with open('user_data.json', 'r') as json_file:
    loaded_user_list = json.load(json_file)

pprint(json.loads(json.dumps(loaded_user_list)))


def main():
    values = loaded_user_list  
    analyser = AnalyseUserList(values)
    analyser.analyse_userlist()


if __name__ == "__main__":
    main()
