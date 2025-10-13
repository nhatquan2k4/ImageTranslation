import pandas as pd
import json
import os

def create_data(json_path):
    """
    Load image-to-translation data from JSON file
    Expected format: [{"id": "...", "image_path": "...", "source_text": "...", "target_text": "..."}]
    """
    with open(json_path, encoding='utf8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Ensure all required fields exist
    required_fields = ['image_path', 'target_text']
    for field in required_fields:
        if field not in df.columns:
            raise ValueError(f"Missing required field: {field}")
    
    # Filter out empty entries
    not_empty = (df['image_path'].str.len() > 0) & (df['target_text'].str.len() > 0)
    df = df[not_empty]
    
    # Convert relative paths to absolute paths if needed
    data_dir = os.path.dirname(json_path)
    df['image_path'] = df['image_path'].apply(
        lambda x: x if os.path.isabs(x) else os.path.join(data_dir, x)
    )
    
    return df

if __name__=="__main__":
    df_train = create_data('C:python/i2t/data/train/train.json')
    df_val = create_data('C:/python/i2t/data/val/val.json')

    print(f"Train size: {len(df_train)}")
    print(f"Validate size: {len(df_val)}")
    print("="*10, "Sample train data", "="*10)
    print(df_train.head())