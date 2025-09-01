import pandas as pd

df = pd.read_csv('PlanetPoints-Datasets - products.csv', encoding='cp1252')  # 不行再试 'latin1' 或 'utf-8-sig'

mult = {'A':3,'B':2,'C':1,'D':0,'E':0,'1':3,'2':2,'3':1,'4':0,'5':0,1:3,2:2,3:1,4:0,5:0}
df['reward points'] = (
    pd.to_numeric(df['price'], errors='coerce')
    * df['impactRating'].astype(str).str.strip().str.upper().map(mult)
).round().astype('Int64')

# df.to_csv('PlanetPoints-Datasets - products_with_points.csv', index=False)
print(df[['price','impactRating','reward points']].head())


# ④ 如需保存，取消下一行的注释
df.to_csv('PlanetPoints-Datasets_products_with_points.csv', index=False)

