from database import db
import pandas as pd

def main():
    # Get all corrections
    print("\n=== All Corrections ===")
    corrections = db.get_corrections()
    print(corrections)
    
    # Get total count
    print("\n=== Total Corrections ===")
    count = db.get_correction_count()
    print(f"Total number of corrections: {count}")
    
    # Get recent corrections (last 5)
    print("\n=== Recent Corrections (last 5) ===")
    recent = db.get_corrections(limit=5)
    print(recent)
    
    # Export to CSV
    print("\n=== Exporting to CSV ===")
    db.export_to_csv("sentiment_corrections_export.csv")
    print("Data exported to sentiment_corrections_export.csv")

if __name__ == "__main__":
    main() 