# Free Qwen API

这是一个免费的通义千问API代理服务，支持通义千问的多个模型，提供与OpenAI API兼容的接口。

## 支持的模型

- qwen-max-latest
- qwen2.5-14b-instruct-1m
- qvq-72b-preview
- qwen-plus-latest
- qwen2.5-vl-72b-instruct

## 环境要求

- Python 3.8+
- FastAPI
- Uvicorn
- 其他依赖见 requirements.txt

## 安装

1. 克隆仓库：
```bash
git clone [仓库地址]
cd free-qwen-api
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置环境变量：
创建 `.env` 文件并添加以下内容：
```
QWEN_API_KEY=你的通义千问API密钥
```

## 运行服务

```bash
python free-qwen-api.py
```
或者使用uvicorn直接运行：
```bash
uvicorn free-qwen-api:app --host 0.0.0.0 --port 8008
```

服务将在 http://localhost:8008 上运行。

## API 端点

### 获取可用模型列表
```
GET /v1/models
```

### 聊天完成
```
POST /v1/chat/completions
```

请求体示例：
```json
{
    "model": "qwen-max-latest",
    "messages": [
        {
            "role": "user",
            "content": "你好"
        }
    ],
    "stream": false
}
```

## 部署建议

本服务是一个后端API服务，需要持续运行并处理请求，不建议部署在Netlify上。建议的部署选项包括：

1. 传统VPS/云服务器（推荐）
   - 阿里云
   - 腾讯云
   - AWS EC2
   - Digital Ocean

2. 容器平台
   - Docker + Kubernetes
   - Google Cloud Run
   - AWS ECS

3. PaaS平台
   - Heroku
   - Railway
   - Render

这些平台都更适合运行持续性的后端服务，可以提供更好的性能和可靠性。

## 注意事项

1. API密钥安全：确保不要将 .env 文件提交到版本控制系统中
2. 请遵守通义千问的使用条款和政策
3. 建议在生产环境中添加适当的速率限制和安全措施

## License

MIT 