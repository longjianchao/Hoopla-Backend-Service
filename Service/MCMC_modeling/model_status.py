import asyncio
import re
from fastapi import APIRouter, WebSocket
from .MCMC_modeling_service import model_log_buffer

mcmc_status_router = APIRouter()

@mcmc_status_router.websocket("/model_status")
async def model_status(websocket: WebSocket):
    """Push modeling logs to the frontend in real time"""
    await websocket.accept()
    print("📡 WebSocket connection established")

    last_pos = 0  # Last read position
    header_sent = False  # Whether the header has been sent
    
    try:
        while True:
            await asyncio.sleep(0.3)
            data = model_log_buffer.getvalue()
            
            if len(data) > last_pos:
                new_output = data[last_pos:]
                lines = new_output.split('\n')
                
                # Process each line
                for line in lines:
                    if not line.strip():
                        continue
                    
                    # Send the header only once
                    if 'Status    | Bounds | Ellipses | Networks | Calls    | f_live | N_eff | log Z' in line and not header_sent:
                        header_sent = True
                        await websocket.send_text(f"{line}\n")
                    elif line.strip():
                        # Define the list of status words
                        status_words = ['Sampling', 'Computing', 'Bounding', 'Finished', 'Stopped']
                        
                        # Check whether this line contains multiple status words
                        status_positions = []
                        for word in status_words:
                            pos = line.find(word)
                            if pos != -1:
                                status_positions.append((pos, word))
                        
                        # If multiple status words are found, split and send them separately
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
                            # Only one status word or none: send the line as-is
                            await websocket.send_text(f"{line}\n")
                
                last_pos = len(data)
    except Exception as e:
        print("WebSocket connection closed:", e)
