import os
from datetime import datetime, timedelta

def delete_old_files(folder_path, older_than_days=14):
    now = datetime.now()
    older_than = now - timedelta(days=older_than_days)

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        
        if os.path.isfile(file_path):
            modified_at = datetime.fromtimestamp(os.path.getmtime(file_path))

            if modified_at < older_than:
                os.remove(file_path)
                print(f'{file} has been deleted')

folder_path = "C:/Users/lukas/Downloads"
delete_old_files(folder_path)
