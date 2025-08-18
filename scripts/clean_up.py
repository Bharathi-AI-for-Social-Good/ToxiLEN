import os
import shutil


def delete_pycache_dirs(root_dir="."):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "__pycache__" in dirnames:
            pycache_path = os.path.join(dirpath, "__pycache__")
            print(f"🧹 Deleting: {pycache_path}")
            try:
                # delete the whole __pycache__ folder
                import shutil
                shutil.rmtree(pycache_path)
            except Exception as e:
                print(f"⚠️ Failed to delete {pycache_path}: {e}")
                

delete_pycache_dirs()