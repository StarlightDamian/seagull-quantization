# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:27:59 2024

@author: awei
(server_fastapi)
"""
import os

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd

from __init__ import path
from utils import utils_database, utils_log, utils_server

app = FastAPI()
current_file_path = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(current_file_path, "../html"))

log_filename = os.path.splitext(os.path.basename(__file__))[0]
logger = utils_log.logger_config_local(f'{path}/log/{log_filename}.log')

# Sample DataFrame
# =============================================================================
# df = pd.DataFrame({
#     'id': [1, 2, 3],
#     'name': ['Alice', 'Bob', 'Charlie'],
#     'score': [85, 90, 99]
# })
# =============================================================================
with utils_database.engine_conn('postgre') as conn:
    df = pd.read_sql("dwd_info_nrtd_portfolio_base", con=conn.engine)

@app.get("/data", response_class=HTMLResponse)
async def get_data(request: Request):
    """
    Return DataFrame data as an HTML table
    """
    columns = df.columns.tolist()
    data = df.to_dict(orient='records')
    return templates.TemplateResponse("chatgpt_style.html", {"request": request, "columns": columns, "data": data})

@app.get("/data/{item_id}")
async def get_data_item(item_id: int):
    """
    Return data for the specified ID
    """
    item = df[df['fund_code'] == item_id]
    if item.empty:
        return {"error": "Item not found"}
    return item.to_dict(orient='records')[0]

if __name__ == "__main__":
    uvicorn.run(app, host="192.168.254.1", port=1024, reload=False)  # host=ipv4 address, you can cmd and ifconfig find it
    ipv4 = utils_server.ipv4_address()
    print(f'ipv4: {ipv4}')
    
    # 192.168.254.1, 10.110.10.56
    # uvicorn main:app --reload
    # python E:/03_software_engineering/github/seagull-quantization/lib/server/server_fastapi.py
    # http://10.110.10.56:1024/data
    # http://192.168.254.1:1024/data
    
    # pip show ngrok
    # cd C:/Users/awei/AppData/Roaming/Python/Python311/site-packages/ngrok
    # ngrok http 1024