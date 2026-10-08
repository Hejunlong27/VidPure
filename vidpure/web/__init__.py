"""vidpure Web 界面 —— 面向非开发用户的本地网页应用。

架构：Python 标准库 http.server（零第三方依赖）提供 REST API 与静态页面；
前端为单文件响应式页面（上传、框选字幕区域、模式选择、进度、结果预览下载）。

启动：vidpure web            （自动打开浏览器 http://127.0.0.1:8787）
      vidpure web --lan     （允许手机同 WiFi 访问）
"""

from .server import serve

__all__ = ["serve"]
