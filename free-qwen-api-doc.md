# Qwen API 代理服务文档

## 概述
本服务是基于FastAPI实现的Qwen API代理，提供与OpenAI兼容的ChatCompletions接口。主要功能是将客户端请求转发至Qwen官方API（`https://chat.qwenlm.ai/api/chat/completions`），并处理响应返回。

## 依赖项
```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import requests
import logging
import json
from datetime import datetime
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field
import uuid
import sys
import os
from dotenv import load_dotenv
```

## 配置说明
### 环境变量
- 从`.env`文件加载环境变量
- 必需变量：`QWEN_API_KEY`（Qwen API密钥）

### 日志配置
- 日志级别：INFO
- 输出：文件(`chat_api.log`)和控制台
- 格式：`%(asctime)s - %(levelname)s - %(message)s`
- 编码：UTF-8

## 全局变量
```python
AVAILABLE_MODELS = set([
    "qwen-max-latest",
    "qwen2.5-14b-instruct-1m",
    "qvq-72b-preview",
    "qwen-plus-latest",
    "qwen2.5-vl-72b-instruct"
])

REMOTE_API_URL = "https://chat.qwenlm.ai/api/chat/completions"

HEADERS = {
    "accept": "*/*",
    "accept-encoding": "identity",
    "authorization": f"Bearer {os.getenv('QWEN_API_KEY')}",
    "content-type": "application/json",
    "origin": "https://chat.qwenlm.ai",
    "referer": "https://chat.qwenlm.ai/c/b01c2449-9e27-47d2-a1d6-4e792a22030c",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)..."
}
```

## 数据模型
### Message模型
```python
class Message(BaseModel):
    role: str = Field(..., description="消息角色：system、user 或 assistant")
    content: Union[str, List[Dict[str, Any]], Dict[str, Any]] = Field(..., description="消息内容")
    chat_type: Optional[str] = Field("artifacts", description="聊天类型")
    extra: Optional[dict] = Field(default_factory=dict, description="额外参数")
    
    def get_content_text(self) -> str:
        # 从不同格式的content中提取文本内容
        ...
```

### ChatCompletionRequest模型
```python
class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="模型名称")
    messages: List[Message] = Field(..., description="对话消息列表")
    stream: bool = Field(False, description="是否使用流式响应")
    temperature: Optional[float] = Field(None, description="温度参数", ge=0, le=2)
    max_tokens: Optional[int] = Field(None, description="最大生成token数")
    stream_options: Optional[dict] = Field(default_factory=dict, description="流式响应选项")
```

## API端点
### GET /v1/models
- **功能**：获取可用模型列表
- **响应格式**：
  ```json
  {
    "object": "list",
    "data": [
      {"id": "model1", "object": "model"},
      {"id": "model2", "object": "model"}
    ]
  }
  ```

### POST /v1/chat/completions
- **功能**：处理聊天补全请求
- **请求处理流程**：
  1. 记录原始请求日志
  2. 验证JSON格式和必需字段
  3. 转换请求格式为Qwen API所需格式
  4. 调用远程API
  5. 处理响应：
     - 非流式响应：构造OpenAI兼容格式
     - 流式响应：处理SSE数据流

- **错误处理**：
  - 400：无效请求（JSON格式错误/缺少必需字段）
  - 400：不支持的模型
  - 500：远程API调用失败
  - 500：内部处理错误

## 运行方式
```bash
uvicorn free-qwen-api:app --host 0.0.0.0 --port 8008
```

## 请求示例
```curl
curl -X POST "http://localhost:8008/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "model": "qwen-max-latest",
  "messages": [
    {"role": "user", "content": "你好"}
  ],
  "stream": false
}'