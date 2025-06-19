# -*- coding: utf-8 -*-
"""
@Date: 2024/10/21 13:07
@Author: Damian
@Email: zengyuwei1995@163.com
@File: utils_vpn.py
@Description: vpn
"""
import subprocess
import sys
import winreg

import adata

# 设置代理,代理是全局设置,代理失效后可重新设置。参数:ip,proxy_url
adata.proxy(is_proxy=True, ip='60.167.21.27:1133')
res_df = adata.stock.info.all_code()
print(res_df)


def set_system_proxy(enable, proxy_server="127.0.0.1:7890"):
    """
    启用或禁用系统代理
    """
    internet_settings = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
        r'Software\Microsoft\Windows\CurrentVersion\Internet Settings',
        0, winreg.KEY_ALL_ACCESS)

    winreg.SetValueEx(internet_settings, 'ProxyEnable', 0, winreg.REG_DWORD, enable)
    if enable:
        winreg.SetValueEx(internet_settings, 'ProxyServer', 0, winreg.REG_SZ, proxy_server)

    # 刷新系统设置
    subprocess.call('taskkill /im iexplore.exe /f', shell=True)
    subprocess.call('ipconfig /flushdns', shell=True)


def enable_proxy():
    """启用代理"""
    set_system_proxy(1)
    print("系统代理已启用")


def disable_proxy():
    """禁用代理"""
    set_system_proxy(0)
    print("系统代理已禁用")


def get_proxy_status():
    """获取当前代理状态"""
    internet_settings = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
        r'Software\Microsoft\Windows\CurrentVersion\Internet Settings',
        0, winreg.KEY_ALL_ACCESS)
    
    try:
        proxy_enable, _ = winreg.QueryValueEx(internet_settings, 'ProxyEnable')
        return bool(proxy_enable)
    except WindowsError:
        return False


def toggle_proxy():
    """切换代理状态"""
    current_status = get_proxy_status()
    if current_status:
        disable_proxy()
    else:
        enable_proxy()


def main():
    if len(sys.argv) < 2:
        print("用法: python proxy_manager.py [enable|disable|toggle|status]")
        return

    action = sys.argv[1].lower()

    if action == "enable":
        enable_proxy()
    elif action == "disable":
        disable_proxy()
    elif action == "toggle":
        toggle_proxy()
    elif action == "status":
        status = "启用" if get_proxy_status() else "禁用"
        print(f"当前系统代理状态: {status}")
    else:
        print("无效的操作。请使用 enable, disable, toggle 或 status。")


if __name__ == "__main__":
    main()
