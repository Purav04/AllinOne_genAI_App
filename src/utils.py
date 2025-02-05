import logging
import os, glob
import sys
try:
    from logger import logger
except:
    from src.logger import logger


from src.exception import CustomException

###




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
