import pandas as pd

import efinance as ef
from __init__ import path

df = ef.stock.get_all_company_performance()
print(df)
df.to_csv(f'{path}/data/ods_acct_incr_efinance_sec_10q.csv', index=False)