# ngrok

## 解决了什么问题

* 当启动了本地服务后，只有本地IP才能看到网页，其他人不能通过互联网访问网页

## 注册

* https://dashboard.ngrok.com/
  * 使用非qq邮箱进行注册
  * 注册后直接登录，不需要登录邮箱扫二维码或者authenticator
  * 登录获取令牌

## 下载exe可执行文件

* https://bin.equinox,io/c/bNyjlmQvY4c/ngrok-v3-stable-windows-amd64.zip

* 放到路径path 下并且解压

  

## 在命令行添加ngrok令牌（authtoken）

* ngrok config add-authtoken 2lKDi7BomW8azX0bdbgVOEF1E1z_5R2f74b76f8LPzNJE3PAi

## 命令行运行

* python E:/03_software_engineering/github/seagull-quantization/lib/server/server_fastapi.py

* ngrok http 192.168.254.1:1024

## 分享URL

* https://ef56-2408-840c-295f-c78d-296d-6e91-24c4-ac5b.ngrok-free.app/data
  * 加上/data后缀，这个是在服务脚本中命名的