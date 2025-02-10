import logging
import os, glob
import sys
import sqlite3, hashlib

from src.exception import CustomException


try:
    from logger import logger
except:
    from src.logger import logger

###



## DELETE operation for folder and file
def delete_files_or_folder(folder_path:list=None, file_path:str=None):
    if folder_path:
        try:
            for folder in folder_path:
                if os.path.isdir(folder):
                    files = glob.glob(folder+"/*")
                    for f in files:
                        os.remove(f)
                    os.removedirs(folder)

                    logger.info(f"folder:{folder} is removed")
        except Exception as e:
            CustomException(e, sys)

    elif file_path:
        try:
            os.remove(file_path)
            logger.info(f"file: {file_path} is removed")
        except Exception as e:
            CustomException(e, sys)
    return 

def check_file_exist(file_name, path):
    try:
        if "/" in file_name:
            file_name = file_name.split("/")[1]

        if os.path.isdir(path):
            files = glob.glob(path+"/*")
            if path+"/"+file_name in files:
                return True
        return False
    except Exception as e:
        CustomException(e, sys)
        return False

## database functions
def create_cache_table(database_file_name="cache_db/Temp_cache.db"):
    try:
        conn = sqlite3.connect(database_file_name)
    except sqlite3.OperationalError as e:
        db_path = os.path.join(os.getcwd(), database_file_name.split("/")[0])
        os.makedirs(db_path, exist_ok=True)
        conn = sqlite3.connect(database_file_name)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cache (
                   key TEXT PRIMARY KEY,
                   value TEXT, -- Store the result as text (adapt as needed)
                   timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                   )
    """)
    conn.commit()
    conn.close()

def get_cache(key, database_file_name="cache_db/Temp_cache.db"):
    key = hashlib.sha256(key.encode()).hexdigest()

    conn = sqlite3.connect(database_file_name)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM cache WHERE key = ?", (key,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return result[0]
    return None

def set_cache(key, value, database_file_name="cache_db/Temp_cache.db"):
    key = hashlib.sha256(key.encode()).hexdigest()

    conn = sqlite3.connect(database_file_name)
    cursor = conn.cursor()
    cursor.execute("REPLACE INTO cache (key, value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()

def get_table_size(database_file_name, table_name):
    try:
        conn = sqlite3.connect(database_file_name)
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
        size_bytes = cursor.fetchone()[0]
        conn.close()
        return size_bytes
    except sqlite3.Error as e:
        CustomException(e, sys)
        return None
    
def clear_cache(database_file_name):
    conn = sqlite3.connect(database_file_name)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM cache")
    conn.commit()
    conn.close()