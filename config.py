import json

def get_chatGPT_api_key():
    with open("secrets.json") as secrets_file:
        my_content = json.load(secrets_file)
        my_secret = my_content["chatGPT_api_key"]
        secrets_file.close()
        return my_secret
