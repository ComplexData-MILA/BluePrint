from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive


import os

def upload_to_drive(file_path, folder_id=None):
    try:
        # Authentication
        gauth = GoogleAuth()
        # Try to load saved client credentials
        gauth.LoadCredentialsFile("credentials.json")
        
        if gauth.credentials is None:
            # Authenticate if credentials don't exist
            gauth.LocalWebserverAuth()
        elif gauth.access_token_expired:
            # Refresh them if expired
            gauth.Refresh()
        else:
            # Initialize the saved creds
            gauth.Authorize()
        
        # Save the current credentials
        gauth.SaveCredentialsFile("credentials.json")
        
        drive = GoogleDrive(gauth)
        # ...rest of your existing code...

        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")

        file_name = os.path.basename(file_path)
        
        # Prepare file metadata
        file_metadata = {'title': file_name}
        if folder_id:
            file_metadata['parents'] = [{'id': folder_id}]

        # Upload file
        gfile = drive.CreateFile(file_metadata)
        gfile.SetContentFile(file_path)
        gfile.Upload()

        print(f"File '{file_name}' uploaded successfully to Google Drive.")
        
    except Exception as e:
        print(f"Error uploading file: {str(e)}")
        raise  # Re-raise the exception to see the full error trace