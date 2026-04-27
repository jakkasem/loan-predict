import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# ขอบเขตการเข้าถึง (Scopes)
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def authenticate_gdrive():
    creds = None
    # ไฟล์ token.json เก็บข้อมูลการยืนยันตัวตนของผู้ใช้
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    
    # ถ้าไม่มี credentials ที่ใช้งานได้ ให้ล็อกอินใหม่
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # คุณต้องดาวน์โหลดไฟล์ credentials.json จาก Google Cloud Console
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    
    return build('drive', 'v3', credentials=creds)

def upload_file(service, file_path, folder_id=None):
    file_name = os.path.basename(file_path)
    
    # --- 1. ค้นหาไฟล์เดิมใน Drive ก่อน ---
    query = f"name = '{file_name}' and trashed = false"
    if folder_id:
        query += f" and '{folder_id}' in parents"
    
    results = service.files().list(q=query, fields="files(id, name)").execute()
    items = results.get('files', [])

    media = MediaFileUpload(file_path, resumable=True)

    if items:
        # --- 2. ถ้าเจอไฟล์เดิม ให้ทำการ Update (ทับไฟล์เดิม) ---
        file_id = items[0]['id']
        file = service.files().update(
            fileId=file_id,
            media_body=media
        ).execute()
        print(f"Updated existing file: {file_name} (ID: {file_id})")
    else:
        # --- 3. ถ้าไม่เจอ ให้ทำการ Create (สร้างใหม่) ---
        file_metadata = {'name': file_name}
        if folder_id:
            file_metadata['parents'] = [folder_id]
        
        file = service.files().create(
            body=file_metadata, 
            media_body=media, 
            fields='id'
        ).execute()
        print(f"Created new file: {file_name} (ID: {file.get('id')})")

if __name__ == '__main__':
    service = authenticate_gdrive()
    
    # รายชื่อไฟล์ที่ต้องการอัพโหลด
    files_to_upload = [
        r"C:\Python_Project\Loan_Data\features.pkl",
        r"C:\Python_Project\Loan_Data\model.pkl",
        r"C:\Python_Project\Loan_Data\label_encoder.pkl"
    ]
    
    # ระบุ Folder ID ของ Google Drive (ถ้ามี)
    # หาได้จาก URL ของโฟลเดอร์ เช่น drive.google.com/drive/folders/ABC123XYZ
    target_folder_id = "19g-mpqQq5Bkh36LFsIE5De3bjF7oWp0s" 

    for file_p in files_to_upload:
        if os.path.exists(file_p):
            upload_file(service, file_p, target_folder_id)
        else:
            print(f"Error: ไม่พบไฟล์ที่ {file_p}")