# -*- coding: utf-8 -*-
"""
@Date: 2025/7/7 16:27
@Author: Damian
@Email: zengyuwei1995@163.com
@File: utils_requirements.py
@Description: 
"""


def mkdir_requirements():
    ...
    # cd D:/03_software_engineering/05_github/seagull
    # pipreqs ./ -y-encoding=utf-8

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
逐条安装 requirements.txt 中的包，遇到安装失败的包则记录并跳过。
"""

import subprocess
import sys

def install_package(pkg: str) -> bool:
    """
    调用 pip 安装单个包，返回是否成功。
    """
    try:
        print(f"Installing {pkg} …")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"  ✅ {pkg} installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ⚠️  Failed to install {pkg}:")
        print(e.stderr.decode().strip())
        return False

def main(requirements_file="requirements.txt"):
    failed = []
    with open(requirements_file, encoding="utf-8") as f:
        for line in f:
            pkg = line.strip()
            # 跳过空行和注释
            if not pkg or pkg.startswith("#"):
                continue
            success = install_package(pkg)
            if not success:
                failed.append(pkg)

    if failed:
        print("\n安装完成，但以下包安装失败（已跳过）：")
        for pkg in failed:
            print(f"  - {pkg}")
    else:
        print("\n✅ 所有包安装成功！")

if __name__ == "__main__":
    # 可通过命令行参数指定 requirements 文件
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r","--requirements", default="requirements.txt",
        help="Requirements 文件路径，默认 requirements.txt"
    )
    args = parser.parse_args()
    main(args.requirements)

