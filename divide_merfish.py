import pandas as pd

df = pd.read_csv('seqfish_plus/Moffitt_and_Bambah-Mukku_et_al_merfish_all_cells.csv')
for i in range(30):
    print(i)
    am = df[df['Animal_ID']==i+1]
    am.to_csv(f'merfish/ca{i}.csv')