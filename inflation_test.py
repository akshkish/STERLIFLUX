import esankhyiki

print("=== SterliFlux CPI DATA TEST ===")

df = esankhyiki.get_data(
    "CPI",
    {
        "base_year": "2024",
        "year": "2026",
        "series": "Current"
    },
    format="df"
)

print("\nCOLUMNS:")
print(df.columns.tolist())

print("\nFIRST 20 ROWS:")
print(df.head(20).to_string())

print("\nSHAPE:")
print(df.shape)