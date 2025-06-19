# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 15:34:32 2024

@author: awei
"""
import socket
import socks
import requests
import yaml
import time
from requests.exceptions import RequestException
path = 'D:/clash/Clash.for.Windows-0.20.16-ikuuu/data/profiles'
def test_connection(server, port, password, cipher):
    # 设置SOCKS5代理
    proxy_url = f"socks5://{server}:{port}"
    proxies = {
        "http": proxy_url,
        "https": proxy_url
    }

    try:
        # 增加连接超时和尝试次数
        for attempt in range(3):
            try:
                # 使用代理访问Google，禁用本地DNS解析
                response = requests.get("https://www.google.com", 
                                        proxies=proxies, 
                                        timeout=30,
                                        verify=False)
                if response.status_code == 200:
                    return True
            except RequestException as e:
                print(f"尝试 {attempt + 1} 失败: {str(e)}")
                time.sleep(5)  # 在重试之前等待5秒
    except Exception as e:
        print(f"连接错误: {str(e)}")
    return False

# 读取配置文件
with open(f'{path}/1708358592002.yml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# 遍历代理列表
for proxy in config['proxies']:
    if proxy['type'] == 'ss':  # 仅测试Shadowsocks代理
        #print(f"测试 {proxy['name']}...")
        if test_connection(proxy['server'], proxy['port'], proxy['password'], proxy['cipher']):
            print(f"成功: {proxy['name']} 可以访问Google")
        else:
            print(f"失败: {proxy['name']} 无法访问Google")

print("测试完成")
# =============================================================================
# import socket
# import socks
# import requests
# import yaml
# path = 'D:/clash/Clash.for.Windows-0.20.16-ikuuu/data/profiles'
# def test_connection(server, port):
#     # 设置SOCKS5代理
#     socks.set_default_proxy(socks.SOCKS5, server, port)
#     socket.socket = socks.socksocket
# 
#     try:
#         # 尝试访问Google
#         response = requests.post("https://www.google.com", timeout=5)
#         if response.status_code == 200:
#             return True
#     except:
#         pass
#     return False
# 
# # 读取配置文件
# with open(f'{path}/1708358592002.yml', 'r', encoding='utf-8') as file:
#     config = yaml.safe_load(file)
# 
# # 遍历代理列表
# for proxy in config['proxies']:
#     if proxy['type'] == 'ss':  # 仅测试Shadowsocks代理
#         #print(f"测试 {proxy['name']}...")
#         if test_connection(proxy['server'], proxy['port']):
#             print(f"成功: {proxy['name']} 可以访问Google")
#         else:
#             print(f"失败: {proxy['name']} 无法访问Google")
# 
# print("测试完成")
# =============================================================================
