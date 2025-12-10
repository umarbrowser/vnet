"""
Utility to download dataset from Google Drive
"""

import os
import gdown
import pandas as pd
from loguru import logger
from pathlib import Path


def download_from_google_drive(file_id: str, output_path: str = "data/dataset.csv"):
    """
    Download file from Google Drive
    
    Args:
        file_id: Google Drive file ID (from shareable link)
        output_path: Where to save the file
    """
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Google Drive direct download URL format
    url = f"https://drive.google.com/uc?id={file_id}"
    
    logger.info(f"Downloading dataset from Google Drive...")
    logger.info(f"File ID: {file_id}")
    logger.info(f"Output: {output_path}")
    
    try:
        # Download file
        gdown.download(url, output_path, quiet=False)
        logger.info(f"✅ Dataset downloaded successfully to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"❌ Failed to download: {e}")
        logger.info("Trying alternative method...")
        
        # Alternative: Manual download instructions
        logger.info("\n" + "="*60)
        logger.info("MANUAL DOWNLOAD INSTRUCTIONS:")
        logger.info("="*60)
        logger.info(f"1. Open this link in your browser:")
        logger.info(f"   https://drive.google.com/uc?export=download&id={file_id}")
        logger.info(f"2. Save the file to: {output_path}")
        logger.info(f"3. Then run your script again")
        logger.info("="*60)
        
        return None


def load_dataset(file_path: str):
    """
    Load dataset from file
    
    Args:
        file_path: Path to dataset file
        
    Returns:
        pandas DataFrame
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    
    logger.info(f"Loading dataset from {file_path}...")
    
    # Try different file formats
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        # Try CSV first
        try:
            df = pd.read_csv(file_path)
        except:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    logger.info(f"✅ Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df


if __name__ == "__main__":
    # Google Drive file ID from the URL
    # URL: https://drive.google.com/file/d/1jWt7YogasRuP_isbRXMhDGVoUbVy-uJ5/view?usp=drivesdk
    file_id = "1jWt7YogasRuP_isbRXMhDGVoUbVy-uJ5"
    
    output_path = "data/veremi_dataset.csv"
    
    # Download dataset
    downloaded_path = download_from_google_drive(file_id, output_path)
    
    if downloaded_path:
        # Load and display info
        df = load_dataset(downloaded_path)
        print(f"\nDataset shape: {df.shape}")
        print(f"\nFirst few rows:")
        print(df.head())
        print(f"\nDataset info:")
        print(df.info())









