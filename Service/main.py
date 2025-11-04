import uvicorn
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Service.extract_background.extract_background_service import extract_background_app
from Service.MCMC_modeling.MCMC_modeling_service import mcmc_app
from Service.MCMC_modeling.model_status import mcmc_status_router

app = FastAPI()

# 添加CORS中间件，允许所有来源和所有方法
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(extract_background_app)
app.include_router(mcmc_app)
app.include_router(mcmc_status_router)

if __name__ == "__main__":
    # 在Linux中，需要使用0.0.0.0作为host
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
    # 在Windows中，使用127.0.0.1作为host
    # uvicorn.run("main:app", host="127.0.0.1", port=8080,reload=True)
