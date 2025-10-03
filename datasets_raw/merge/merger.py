import pandas as pd

# Memuat dataset
df1 = pd.read_csv('../datasets_fomo_6000_nolabel.csv')
df2 = pd.read_csv('../datasets_fomo_4000_nolabel.csv')

# Menggabungkan dataframe
merged_df = pd.concat([df1, df2], ignore_index=True)

# Memeriksa duplikat
duplicate_rows = merged_df[merged_df.duplicated()]
print("Jumlah baris duplikat:", duplicate_rows.shape[0])

# Menghapus duplikat (jika ada)
merged_df_unique = merged_df.drop_duplicates()

# Menampilkan info dari dataframe yang digabungkan
print("\nInfo dari dataframe yang digabungkan setelah menghapus duplikat:")
merged_df_unique.info()

# Menyimpan dataframe yang digabungkan ke file CSV baru
merged_df_unique.to_csv('merged_fomo_dataset.csv', index=False)

print("\nDataset yang digabungkan disimpan ke 'merged_fomo_dataset.csv'")