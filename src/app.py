import uvicorn
from uvicorn.config import LOGGING_CONFIG
from fastapi import FastAPI
from src.routes.main_router import router as main_router
from src.routes.skill_router import router as skill_router
from src.routes.complevel_router import router as complevel_router
from src.routes.profile_router import router as profile_router
from src.setup import setup

# Initialize app
app = FastAPI()

# Initialize resources
embedding_functions, skilldb, reranker, domains, db = setup()

# Store resources in app's state so they can be accessed in views
app.state.EMBEDDING_FUNCTIONS = embedding_functions
app.state.SKILLDB = skilldb
app.state.RERANKER = reranker
app.state.DOMAINS = domains
app.state.DB = db

# Register routes
app.include_router(main_router)
app.include_router(skill_router)
app.include_router(complevel_router)
app.include_router(profile_router)

if __name__ == "__main__":
    LOGGING_CONFIG["formatters"]["access"]["fmt"] = '%(asctime)s %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
    uvicorn.run(app, host="0.0.0.0", port=8000)
    