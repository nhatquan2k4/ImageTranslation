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
    
    # Handle image paths - check if they exist as-is or need relative path conversion
    def fix_image_path(image_path):
        # If path exists as-is, use it
        if os.path.exists(image_path):
            return image_path
        
        # If it's already absolute but doesn't exist, try making it relative from project root
        if os.path.isabs(image_path) or image_path.startswith('i2t_dataset'):
            # Remove any leading path components and use as relative to current directory
            if 'i2t_dataset' in image_path:
                # Extract path from i2t_dataset onwards
                parts = image_path.split(os.sep)
                if 'i2t_dataset' in parts:
                    idx = parts.index('i2t_dataset')
                    relative_path = os.path.join(*parts[idx:])
                    if os.path.exists(relative_path):
                        return relative_path
        
        # Fallback: join with data_dir
        data_dir = os.path.dirname(json_path)
        return os.path.join(data_dir, os.path.basename(image_path))
    
    df['image_path'] = df['image_path'].apply(fix_image_path)
    
    return df

if __name__=="__main__":
    df_train = create_data('C:python/i2t/data/train/train.json')
    df_val = create_data('C:/python/i2t/data/val/val.json')

    print(f"Train size: {len(df_train)}")
    print(f"Validate size: {len(df_val)}")
    print("="*10, "Sample train data", "="*10)
    print(df_train.head())