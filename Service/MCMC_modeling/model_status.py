import asyncio
import re
from fastapi import APIRouter, WebSocket
from .MCMC_modeling_service import model_log_buffer

mcmc_status_router = APIRouter()

@mcmc_status_router.websocket("/model_status")
async def model_status(websocket: WebSocket):
    """实时推送建模日志给前端"""
    await websocket.accept()
    print("📡 WebSocket 连接建立")

    last_pos = 0  # 上次读取位置
    header_sent = False  # 标记表头是否已发送
    
    try:
        while True:
            await asyncio.sleep(0.3)
            data = model_log_buffer.getvalue()
            
            if len(data) > last_pos:
                new_output = data[last_pos:]
                lines = new_output.split('\n')
                
                # 处理每一行
                for line in lines:
                    if not line.strip():
                        continue
                    
                    # 表头只发送一次
                    if 'Status    | Bounds | Ellipses | Networks | Calls    | f_live | N_eff | log Z' in line and not header_sent:
                        header_sent = True
                        await websocket.send_text(f"{line}\n")
                    elif line.strip():
                        # 定义状态词列表
                        status_words = ['Sampling', 'Computing', 'Bounding', 'Finished', 'Stopped']
                        
                        # 检查这一行是否包含多个状态词
                        status_positions = []
                        for word in status_words:
                            pos = line.find(word)
                            if pos != -1:
                                status_positions.append((pos, word))
                        
                        # 如果找到多个状态词，分割并分别发送
                        if len(status_positions) > 1:
                            for i in range(len(status_positions)):
                                start_pos = status_positions[i][0]
                                if i < len(status_positions) - 1:
                                    end_pos = status_positions[i + 1][0]
                                    status_line = line[start_pos:end_pos].strip()
                                else:
                                    status_line = line[start_pos:].strip()
                                
                                if status_line:
                                    await websocket.send_text(f"{status_line}\n")
                        else:
                            # 只有一个状态词或没有状态词，直接发送
                            await websocket.send_text(f"{line}\n")
                
                last_pos = len(data)
    except Exception as e:
        print("WebSocket连接关闭:", e)
