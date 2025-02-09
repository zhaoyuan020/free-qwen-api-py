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

# 加载环境变量
load_dotenv()

# 配置日志，设置编码为utf-8
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("chat_api.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

app = FastAPI()

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
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36 Edg/132.0.0.0"
}

class Message(BaseModel):
    role: str = Field(..., description="消息角色：system、user 或 assistant")
    content: Union[str, List[Dict[str, Any]], Dict[str, Any]] = Field(..., description="消息内容")
    chat_type: Optional[str] = Field("artifacts", description="聊天类型")
    extra: Optional[dict] = Field(default_factory=dict, description="额外参数")

    def get_content_text(self) -> str:
        """从不同格式的content中提取文本内容"""
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            # 处理列表格式，提取所有text字段
            texts = []
            for item in self.content:
                if isinstance(item, dict) and "text" in item:
                    texts.append(item["text"])
            return "\n".join(texts)
        elif isinstance(self.content, dict) and "text" in self.content:
            # 处理字典格式
            return self.content["text"]
        return str(self.content)

class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="模型名称")
    messages: List[Message] = Field(..., description="对话消息列表")
    stream: bool = Field(False, description="是否使用流式响应")
    temperature: Optional[float] = Field(None, description="温度参数", ge=0, le=2)
    max_tokens: Optional[int] = Field(None, description="最大生成token数")
    stream_options: Optional[dict] = Field(default_factory=dict, description="流式响应选项")

@app.get("/v1/models")
async def list_models():
    """获取可用模型列表"""
    models = [{"id": model, "object": "model"} for model in AVAILABLE_MODELS]
    return {
        "object": "list",
        "data": models
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """接收原始请求并记录"""
    try:
        # 获取并记录原始请求数据
        body = await request.body()
        body_text = body.decode()
        logging.info(f"Raw request body: {body_text}")
        
        # 尝试解析JSON
        try:
            data = json.loads(body_text)
        except json.JSONDecodeError:
            logging.error("Invalid JSON in request body")
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "Invalid JSON format", "type": "invalid_request_error"}}
            )
        
        # 验证必需字段
        if "model" not in data:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "model field is required", "type": "invalid_request_error"}}
            )
        
        if "messages" not in data or not isinstance(data["messages"], list) or not data["messages"]:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": "messages field must be a non-empty array", "type": "invalid_request_error"}}
            )
        
        # 将验证后的数据转换为我们的格式
        chat_request = ChatCompletionRequest(
            model=data["model"],
            messages=[Message(**msg) for msg in data["messages"]],
            stream=data.get("stream", False),
            temperature=data.get("temperature"),
            max_tokens=data.get("max_tokens"),
            stream_options=data.get("stream_options", {})
        )
        
        # 检查模型是否支持
        if chat_request.model not in AVAILABLE_MODELS:
            return JSONResponse(
                status_code=400,
                content={"error": {"message": f"Model {chat_request.model} not found", "type": "invalid_request_error"}}
            )

        # 构造远程API请求体
        messages_data = []
        for msg in chat_request.messages:
            message_dict = {
                "role": msg.role,
                "content": msg.get_content_text(),
                "chat_type": "artifacts",
                "extra": {}
            }
            messages_data.append(message_dict)

        remote_data = {
            "stream": chat_request.stream,
            "chat_type": "artifacts",
            "model": chat_request.model,
            "messages": messages_data,
            "temperature": chat_request.temperature if chat_request.temperature is not None else 0.7,
            "max_tokens": chat_request.max_tokens if chat_request.max_tokens is not None else 2048,
            "session_id": str(uuid.uuid4()),
            "chat_id": str(uuid.uuid4()),
            "id": str(uuid.uuid4()),
            "size": "960*960"
        }

        try:
            response = requests.post(REMOTE_API_URL, headers=HEADERS, json=remote_data, stream=chat_request.stream)
            response.raise_for_status()
            
            if not chat_request.stream:
                try:
                    remote_data = response.json()
                    logging.info(f"Non-stream response: {json.dumps(remote_data, ensure_ascii=False)}")
                    content = ""
                    if "choices" in remote_data and len(remote_data["choices"]) > 0:
                        message = remote_data["choices"][0].get("message", {})
                        content = message.get("content", "")
                    
                    return {
                        "id": f"chatcmpl-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        "object": "chat.completion",
                        "created": int(datetime.now().timestamp()),
                        "model": chat_request.model,
                        "choices": [{
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": content
                            },
                            "finish_reason": "stop"
                        }],
                        "usage": {
                            "prompt_tokens": -1,
                            "completion_tokens": -1,
                            "total_tokens": -1
                        }
                    }
                except Exception as e:
                    logging.error(f"Failed to process response: {e}")
                    logging.error(f"Raw response: {response.text}")
                    return JSONResponse(
                        status_code=500,
                        content={"error": {"message": f"Failed to process response: {str(e)}", "type": "internal_error"}}
                    )

            async def generate():
                last_text = ""
                try:
                    for line in response.iter_lines():
                        if not line:
                            continue
                        line_text = line.decode('utf-8')
                        
                        if line_text.startswith('data: '):
                            data_text = line_text[6:]
                            if data_text == "[DONE]":
                                yield f"data: [DONE]\n\n".encode('utf-8')
                                continue
                            
                            try:
                                data = json.loads(data_text)
                                if "choices" in data and len(data["choices"]) > 0:
                                    current_text = data["choices"][0].get("delta", {}).get("content", "")
                                    
                                    if current_text.startswith(last_text) and len(last_text) > 0:
                                        delta_text = current_text[len(last_text):]
                                    else:
                                        delta_text = current_text
                                    
                                    if delta_text:
                                        chunk = {
                                            "id": f"chatcmpl-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                                            "object": "chat.completion.chunk",
                                            "created": int(datetime.now().timestamp()),
                                            "model": chat_request.model,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {"content": delta_text},
                                                "finish_reason": None
                                            }]
                                        }
                                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode('utf-8')
                                    
                                    last_text = current_text
                            except json.JSONDecodeError as e:
                                logging.error(f"Failed to parse streaming response: {e}, line: {data_text}")
                                continue
                except Exception as e:
                    logging.error(f"Streaming error: {e}")
                    yield f"data: {{\"error\": \"{str(e)}\"}}\n\n".encode('utf-8')

            return StreamingResponse(generate(), media_type="text/event-stream")
        
        except requests.exceptions.RequestException as e:
            logging.error(f"Remote API request failed: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": {"message": f"Failed to call remote API: {str(e)}", "type": "internal_error"}}
            )
    
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "internal_error"}}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)