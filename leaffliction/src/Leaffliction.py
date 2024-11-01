import os
from urllib.request import urlretrieve
import zipfile
import Shared


def _download_and_extract_resources():
    print("Downloading leaves.zip...", end= "", flush=True)
    url = 'https://cdn.intra.42.fr/document/document/17458/leaves.zip'
    filename = 'leaves.zip'
    filepath = Shared.LEAVES_DIRECTORY + "/" + filename
    
    urlretrieve(url, filepath)
    print("OK.", end= " ", flush=True)
    
    print("Extracting...", end= "", flush=True)
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(Shared.LEAVES_DIRECTORY)
    
    print("OK.")
    if os.path.exists(filepath):
        os.remove(filepath)
    pass

def change_directory(path):
    Shared.BASE_DIRECTORY = path
    Shared.RESSOURCE_DIRECTORY = f"{Shared.BASE_DIRECTORY}/ressources"
    Shared.LEAVES_DIRECTORY = f"{Shared.RESSOURCE_DIRECTORY}/leaves"
    
def init_project():
    dir_path = os.getcwd()
    change_directory(dir_path)

    if os.path.isdir(Shared.RESSOURCE_DIRECTORY) == False:
        os.mkdir(Shared.RESSOURCE_DIRECTORY)
        print("Created missing 'ressource' directory.")
    if os.path.isdir(Shared.LEAVES_DIRECTORY) == False:
        os.mkdir(Shared.LEAVES_DIRECTORY)
        print("Created missing 'leaves' directory.")
        _download_and_extract_resources()
        
if __name__ == '__main__':
    init_project()