import os
import requests
from pathlib import Path

tess_orbit_time_url = 'https://tess.mit.edu/public/files/TESS_orbit_times.csv'

tess_FFI_time_url = 'https://tess.mit.edu/public/files/TESS_FFI_observation_times.csv'


def get_local_file_path():
    'Determine a consistent local file path'
    local_dir = Path(__file__).resolve().parent / ".cache"
    os.makedirs(local_dir, exist_ok=True)
    return local_dir / 'tess_orbit_time.csv'



def download_csv_file(url=tess_orbit_time_url):
    'Downloads the file from the URL and saves it locally'
    local_file_path = get_local_file_path()
    
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()

        with open(local_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return str(local_file_path)

    except requests.exceptions.RequestException as e:
        print(f"Could not download file: {e}")
        if local_file_path.exists():
            return str(local_file_path)
        else:
            raise FileNotFoundError(f"No local or remote data found.")
