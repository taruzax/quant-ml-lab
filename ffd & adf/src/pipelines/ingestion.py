import polars as pl
import pathlib
import yaml
import datetime
from ffd.src.data.loader import load_market_data
from ffd.src.data.transform import apply_features

def load_tickers(config_path):
    path = pathlib.Path(config_path)
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    tickers = [item['ticker'] for item in data]
    return tickers
        

def save_model_data(df: pl.DataFrame, directory: str = "data/raw", filename: str = "model_data"):
    path = pathlib.Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    full_path = path / f"{filename}_{timestamp}.parquet"
    
    df.write_parquet(full_path)
    print(f"Data saved successfully to {full_path}")

if __name__ == "__main__":
    config_path = "config/tickers.yaml"
    tickers = load_tickers(config_path)
    raw_df = load_market_data(tickers=tickers, interval="60m", start="2025-01-01")
    
    # 2. Transform Data
    model_df = apply_features(raw_df, d_value=0.4, threshold=0.001)
    
    # 3. Save Data
    # Specify the target directory relative to your project root
    save_model_data(model_df, directory="data/raw", filename="model_data")
    
    print("Pipeline execution complete!")
