# UPDATE: Replace placeholder paths with your actual paths.
import json
import pyarrow as pa
from datetime import datetime

import pyarrow as pa

def read_arrow_file_correctly(file_path):
    """Read Arrow IPC format file correctly"""
    try:
        with open(file_path, "rb") as f:
            # Use RecordBatchStreamReader to read Arrow IPC format
            reader = pa.ipc.RecordBatchStreamReader(f)
            table = reader.read_all()
            return table
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

# Read your file correctly
file_path = 'data/public_drive_datasets/Ego3D-Bench/test/data-00000-of-00001.arrow'
table = read_arrow_file_correctly(file_path)

def arrow_to_json_complete(table, output_json_path):
    """
    Convert Arrow table completely to JSON format
    """
    # Convert to pandas DataFrame
    df = table.to_pandas()

    # Handle possible complex data types (e.g. dates, lists, etc.)
    def convert_to_serializable(obj):
        """Convert non-JSON-serializable objects to basic types"""
        if isinstance(obj, (datetime, pa.Timestamp)):
            return obj.isoformat()
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
            return list(obj)
        else:
            return obj

    # Build JSON data structure
    json_data = {
        "metadata": {
            "conversion_date": datetime.now().isoformat(),
            "source_format": "arrow",
            "num_records": len(df),
            "columns": df.columns.tolist(),
            "shape": df.shape
        },
        "records": []
    }

    # Convert each record
    for idx, row in df.iterrows():
        record = {}
        for col in df.columns:
            try:
                record[col] = convert_to_serializable(row[col])
            except Exception as e:
                record[col] = str(row[col])  # Convert to string as last resort
        json_data["records"].append(record)

    # Save as JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)

    return json_data

# Execute conversion
output_path = "data/Ego3d-Bench/ego3d_bench_converted.json"
json_result = arrow_to_json_complete(table, output_path)
print(f"JSON file saved to: {output_path}")
