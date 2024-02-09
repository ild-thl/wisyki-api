import uvicorn
from fastapi import FastAPI
from src.routes.main_router import router as main_router
from src.routes.skill_router import router as skill_router
from src.routes.complevel_router import router as complevel_router
from src.setup import setup

app = FastAPI()

# Initialize resources
embedding_functions, skilldbs, reranker, db = setup()

# Store resources in app's state so they can be accessed in views
app.state.EMBEDDING_FUNCTIONS = embedding_functions
app.state.SKILLDBS = skilldbs
app.state.RERANKER = reranker
app.state.DB = db

# Register routes
app.include_router(main_router)
app.include_router(skill_router)
app.include_router(complevel_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)