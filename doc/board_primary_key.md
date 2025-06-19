df.drop_duplicates('board_primary_key',keep='first')[['board_type','is_limit_up_prev','is_limit_down_prev','is_flat_price_prev']]
Out[23]: 
       board_type  is_limit_up_prev  is_limit_down_prev  is_flat_price_prev
0              指数             False               False               False
65             主板             False               False               False
391            主板              True               False               False
625            主板              True               False                True
715            ST             False               False               False
716            ST              True               False               False
732            ST             False                True               False
1106           主板             False                True               False
2364           主板             False               False                True
13976          主板              True                True                True
17492          ST              True               False                True
18431          ST              True                True               False
20604          ST             False               False                True
20621          ST             False                True                True
26781          主板             False                True                True
34984          指数             False                True               False
34985          指数              True               False               False
42697          主板              True                True               False
109753        ETF             False               False               False
110150        ETF              True                True                True
110287        ETF              True               False               False
110438        ETF             False                True               False
110579        ETF              True                True                True
110580        ETF             False               False               False
112089        ETF             False               False                True
118986        ETF              True               False               False
130091        创业板             False               False               False
130428        创业板              True               False               False
130499        创业板             False                True               False
130805        创业板              True                True                True
138605        创业板             False               False                True
141739        创业板              True               False                True
143284         ST              True                True                True
146712        创业板             False                True                True
235067        北交所             False               False               False
235180        北交所              True               False                True
235181        北交所              True               False               False
376614        科创板             False               False               False
379019        科创板             False                True               False
379020        科创板              True               False               False
381217        科创板              True                True                True
399101        科创板             False               False                True
402780        科创板              True               False                True
414395        北交所             False                True               False
417962        北交所             False               False                True
427697        北交所              True                True                True
427785        北交所             False                True                True

board_primary_key
02dd62366de63b3fae4d2c7779ef3f09        26
065b083ba1d8b4a8cfc89008f0df98eb     48850
14c515ab6ec5fd131292c6fed809d164        26
1fccc0a358e39f7029daab2e937bad1e       204
22dad39d8240cfc2b2eb9290514d23a6     36404
24404cd23312afe2449ae25c62b5214e      5005
2a388329f7b0620995ffc4bee07241d1       430
2a6a691b9f5ba45f1c23ef175e55a59b         1
3f4fb1058a407b4748a96225372eefb3         1
41aa9fda623b7e9755572a53538e50a8     32302
48a6c72e8a65823c06ee048da724e931        18
5099248f6007a11fc5e1a483d94a9f10         2
50ab7e5bc5ab930e4f1128deba72ecc2       126
52b4837c96c1b65af288b915d721c658     14624
5958e5fbc2f15e82a872ffb8ef622a76        35
59bc9f2d011040cc784d04566aae301e         5
5a66a11f02a7eb252eb5ca9158a3f98d        39
613ba9f901aaa50f0f98b6308c63d7f6         1
638a482b1fa9675d1646695e1bf3ce53         9
66da19668f700a2eacf1089338adcd66        37
82d3dceca8d229d702ee29cd01f1e200     84645
85122f7f3f35f8a0667b74aa882bd327       568
87e6e27be8998b9b7fac9f65d4eff674        17
88b813a5105d9328b425c6a5dc697b9b        27
88c2e544c569a3373e431b67313b59bd         7
88e358b3cdf6b58149c58bef7dcdd2ef       229
8a73ba4856082710b0737fd93e565da9         6
900ba22a7a0c7735ec2847ac53476d56         2
972dc79968bbbc8a495db16d5b035e36         5
97eb3951adf3fb0c44ead3716b39daa9        26
9f045798ef76188070e6785dd35c21d0        29
b9971522bcab9571a667c7652dfc6cdc       254
baeb34c74bd3665e06561a950affa615         1
bb1abf41f104eb612c07bf5b2ebf0bd7        28
c038d2e98e0a65b6793e0ba99b55dda9    195697
c33ec21b8792799cf7b951dd9d94bf27         3
cb33b3aa87338f159357d0129a60bf99        56
e036ca6b2d8cf2b91464a072c075a9ed        84
e48084997e08d1c2bf3c6d5d62ba25a0      3775
e4c8600873db27c5fc07a7eddcd2afd5       787
e702b5c9ce76514b9ce700bef28de414        47
e85111d689d1664a4a59cd553201872f         4
ed71fb27d2576fcdb9a3c8865ed28384         1
f1b5b37b6e1e2270b9f628b1715cc308        32
f68c762f7762d26fd86c42d97999ab5c        10
fc009d507bcde4c86a42256fcba65859      3130
fc70674c21e46b6aa3bf2ce108a26e3f       277
dtype: int64



df.drop_duplicates('board_primary_key',keep='first')[['board_type','price_limit_rate','is_limit_up_prev','is_limit_down_prev']]
Out[10]: 
       board_type  price_limit_rate  is_limit_up_prev  is_limit_down_prev
0              指数              0.10             False               False
65             主板              0.10             False               False
391            主板              0.10              True               False
715            ST              0.05             False               False
716            ST              0.05              True               False
732            ST              0.05             False                True
1106           主板              0.10             False                True
13845          主板              0.10              True                True
18366          ST              0.05              True                True
34790          指数              0.10             False                True
34791          指数              0.10              True               False
42512          主板              0.10              True                True
109298        ETF              0.10             False               False
110838        ETF              0.10              True                True
111524        ETF              0.10              True               False
111675        ETF              0.10             False                True
111816        ETF              0.20              True                True
111817        ETF              0.20             False               False
121085        ETF              0.20              True               False
130480        创业板              0.20             False               False
130611        创业板              0.20              True                True
130804        创业板              0.20              True               False
130832        创业板              0.20             False                True
143285         ST              0.05              True                True
234938        北交所              0.30             False               False
235175        北交所              0.30              True               False
376484        科创板              0.20             False               False
379832        科创板              0.20              True               False
380040        科创板              0.20             False                True
381437        科创板              0.20              True                True
414524        北交所              0.30             False                True
427710        北交所              0.30              True                True



result_df.reset_index(drop=True).sort_values(by=['board_type','price_limit_rate','is_limit_up_prev','is_limit_down_prev'],ascending=False)
Out[23]: 
   board_type  price_limit_rate  is_limit_up_prev  is_limit_down_prev     num
17        科创板              0.20              True                True      17
1         科创板              0.20              True               False      57
13        科创板              0.20             False                True       5
30        科创板              0.20             False               False   36406
10         指数              0.10              True               False       1
25         指数              0.10             False                True       1
24         指数              0.10             False               False   32302
22        北交所              0.30              True                True       4
31        北交所              0.30              True               False     209
20        北交所              0.30             False                True      11
26        北交所              0.30             False               False   14630
15        创业板              0.20              True                True      37
16        创业板              0.20              True               False     287
21        创业板              0.20             False                True      27
14        创业板              0.20             False               False   84672
5          主板              0.10              True                True      32
9          主板              0.10              True                True      84
19         主板              0.10              True               False    3359
27         主板              0.10             False                True     813
0          主板              0.10             False               False  195823
11         ST              0.05              True                True       3
23         ST              0.05              True                True      28
3          ST              0.05              True               False     607
18         ST              0.05             False                True     459
7          ST              0.05             False               False    5023
12        ETF              0.20              True                True       7
28        ETF              0.20              True               False       1
4         ETF              0.20             False               False    3775
6         ETF              0.10              True                True      47
8         ETF              0.10              True               False      35
29        ETF              0.10             False                True      26
2         ETF              0.10             False               False   49104



def board(df):
    df['num'] = df.shape[0]
    df = df.drop_duplicates('board_primary_key',keep='first')
    return df[['board_type','price_limit_rate','is_limit_up_prev','is_limit_down_prev','num']]

result_df = df.groupby('board_primary_key').apply(board).reset_index(drop=True)
result_df.reset_index(drop=True).sort_values(by=['board_type','price_limit_rate'],ascending=False)
E:\temp\ipykernel_7832\197234291.py:6: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
  result_df = df.groupby('board_primary_key').apply(board).reset_index(drop=True)
Out[27]: 
  board_type  price_limit_rate  is_limit_up_prev  is_limit_down_prev     num
2        科创板              0.20             False               False   36485
4         指数              0.10             False               False   32304
7        北交所              0.30             False               False   14854
1        创业板              0.20             False               False   85023
0         主板              0.10             False               False  200111
3         ST              0.05             False               False    6120
5        ETF              0.20              True                True    3783
6        ETF              0.10             False               False   49212