import requests

class CloudSQLite:
    def __init__(self, ip_address, connection_name):
        self.ip_address = ip_address
        self.connection_name = connection_name

    def execute_query(self, query):
        url = f"http://{self.ip_address}/{self.connection_name}/query"
        response = requests.post(url, json={"query": query})
        return response.json()

# Beispiel für die Verwendung
cloud_sqlite = CloudSQLite('34.159.175.13', "newvalueai:europe-west3:test1")

create_table_query = "CREATE TABLE IF NOT EXISTS example_table (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)"
response = cloud_sqlite.execute_query(create_table_query)
print(response)  # Hier erhalten Sie die Antwort von der API

insert_query = "INSERT INTO example_table (name, age) VALUES ('John Doe', 30)"
response = cloud_sqlite.execute_query(insert_query)
print(response)  # Hier erhalten Sie die Antwort von der API


connect1 = CloudSQLite()
