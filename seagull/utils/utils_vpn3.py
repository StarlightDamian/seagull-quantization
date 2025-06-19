# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 15:25:25 2024

@author: awei
"""
import subprocess
import os
path = 'D:/clash/Clash.for.Windows-0.20.16-ikuuu/data/profiles'
def ping_test(ip):
    response = os.system(f"ping -c 1 {ip}")
    if response == 0:
        print(f"{ip} is reachable")
        return True
    else:
        print(f"{ip} is not reachable")
        return False

# 测试是否可以访问 Google DNS
ping_test("8.8.8.8")

def start_vpn(config_file):
    try:
        subprocess.run(["sudo", "openvpn", "--config", config_file], check=True)
        print("VPN started")
    except subprocess.CalledProcessError as e:
        print("Failed to start VPN:", e)

# 停止 VPN
def stop_vpn():
    try:
        subprocess.run(["sudo", "killall", "openvpn"], check=True)
        print("VPN stopped")
    except subprocess.CalledProcessError as e:
        print("Failed to stop VPN:", e)
def start_vpn_with_admin(config_file):
    # 在 Windows 中使用 runas 来以管理员权限运行
    command = f'runas /user:administrator "openvpn --config {config_file}"'
    try:
        subprocess.run(command, shell=True, check=True)
        print("VPN started with admin privileges")
    except subprocess.CalledProcessError as e:
        print("Failed to start VPN:", e)
# 启动 VPN，假设使用 OpenVPN 配置文件
vpn_config = "f'{PATH}/1708358592002.yml'"
start_vpn_with_admin(vpn_config)

start_vpn(vpn_config)

# 之后可以测试 Google 是否能连通
if ping_test("8.8.8.8"):
    print("Successfully connected to the VPN and Google is reachable.")
else:
    print("Failed to connect to Google.")

vpn_servers = [
    {"name": "🇭🇰 香港W01", "server": "o8x09-g01.hk01-ae5.entry.v50307shvkaa.art", "port": 19272},
    {"name": "🇯🇵 日本W01", "server": "so8ir-g01.jp01-ae5.entry.v50307shvkaa.art", "port": 473}
    # 更多服务器配置
]

for server in vpn_servers:
    print(f"Testing VPN server: {server['name']} at {server['server']}:{server['port']}")
    # 配置 VPN 连接和测试 Google 连接
    if ping_test("8.8.8.8"):
        print(f"VPN {server['name']} connected successfully!")
    else:
        print(f"VPN {server['name']} failed to connect.")
