您的需求非常明确，我理解您想要一个Python脚本来管理Clash的系统代理，以便在运行不同脚本时可以灵活地切换VPN策略。这是一个很实用的想法，特别是在需要同时访问内网和外网资源的开发环境中。我会为您提供一个脚本示例，它可以实现这个功能。



```python
import subprocess
import sys
import winreg

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

```

这个脚本提供了以下功能：

1. 启用系统代理
2. 禁用系统代理
3. 切换代理状态（如果当前是启用状态则禁用，反之亦然）
4. 查询当前代理状态

使用方法：

1. 启用代理：`python proxy_manager.py enable`
2. 禁用代理：`python proxy_manager.py disable`
3. 切换代理状态：`python proxy_manager.py toggle`
4. 查询代理状态：`python proxy_manager.py status`

这个脚本通过修改 Windows 注册表来控制系统代理。它假设 Clash 的默认代理地址是 127.0.0.1:7890，如果您的 Clash 配置不同，可以在 `set_system_proxy` 函数中修改 `proxy_server` 参数。

要在不同的脚本中使用这个代理管理器，您可以：

1. 在脚本开始时调用 `python proxy_manager.py enable` 来启用代理。
2. 在脚本结束时调用 `python proxy_manager.py disable` 来禁用代理。
3. 或者，您可以将这个脚本作为一个模块导入到您的其他 Python 脚本中，直接调用其中的函数。

例如：

```python
import subprocess

def run_with_proxy(script_name):
    subprocess.call('python proxy_manager.py enable', shell=True)
    try:
        subprocess.call(f'python {script_name}', shell=True)
    finally:
        subprocess.call('python proxy_manager.py disable', shell=True)

# 使用示例
run_with_proxy('需要外网的脚本.py')
```

这样，您就可以根据不同脚本的需求灵活地控制系统代理了。

请注意，这个脚本需要管理员权限来修改系统设置。此外，它是为 Windows 系统设计的。如果您使用其他操作系统，可能需要相应地修改代码。

如果您需要对脚本进行任何修改或有任何其他问题，请随时告诉我。