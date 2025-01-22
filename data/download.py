import requests
import os

# URL of the file to download
url = "https://storage.googleapis.com/kaggle-data-sets/1573099/2589570/compressed/part-00000-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250121%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250121T003413Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=8b7467c04a83210223546cfe8bbff8b36eaf44d60c1fc5722bae6dd21506f73fddb5fc50030aafd6a56553b282b540be7fd5a50842561ecea2548794b835683046f865ebb35ad71210513d6d45e30dba3397e185ae9d2a6cb1b79c7b1f1ed43c6f5b9ba553bd84a629cba2aabe334bf4c50b60a77a890399d89369f04dc9ee289b6274c46b11f1e303aeef7164de7ae5803154d91ae924a70a0d0a4e82e092c8fc1a412905b38d6b895f072284a714e8ce826ed7f31fb1eeea419e477762291ea1d019c5da5bef7d4c7771d730ee04afab1384be29f68e80ad655971e52f94b1ec14f397f956acfee7067f525e818ffa996dcc634224088ea7d01ad823f931f3"

# Local filename to save the file
local_filename = "part-00000.snappy.parquet.zip"

# Make the request and save the file
try:
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check for HTTP request errors

    with open(local_filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"File downloaded successfully and saved as {local_filename}")

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
