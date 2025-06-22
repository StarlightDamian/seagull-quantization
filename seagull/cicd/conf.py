import os, sys
sys.path.insert(0, os.path.abspath('../..'))

extensions = [
    'autoapi.extension',
]

# autoapi 配置：指向你的源码目录
autoapi_type = 'python'
autoapi_dirs = ['..']     # 从 docs/ 的上一级开始搜索 .py
autoapi_root = 'api'      # 在 _build/html/api/ 下生成文档
autoapi_options = [
    'members', 'undoc-members', 'show-module-summary',
    'show-inheritance',
]

html_theme = 'sphinx_rtd_theme'

# sphinx-quickstart docs
# y
# seagull
# Damian
# v0.0.1
# zh-cn
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-option_emphasise_placeholders