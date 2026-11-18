import asyncio
import uvicorn
from uvicorn.config import LOGGING_CONFIG
from fastapi import FastAPI
from src.routes.main_router import router as main_router
from src.routes.skill_router import router as skill_router
from src.routes.complevel_router import router as complevel_router
from src.routes.metadata_router import router as metadata_router
from src.setup import setup
from src.models.metadata_job_store import MetadataJobStore

# Initialize app
app = FastAPI()

# Initialize resources
embedding_function, skilldb, domains, db, skill_collection = setup()

# Store resources in app's state so they can be accessed in views
app.state.EMBEDDING_FUNCTION = embedding_function
app.state.SKILLDB = skilldb
app.state.DOMAINS = domains
app.state.DB = db
app.state.SKILL_COLLECTION = skill_collection
app.state.METADATA_STORE = MetadataJobStore.from_env()


@app.on_event("startup")
async def initialize_metadata_store():
    await asyncio.to_thread(app.state.METADATA_STORE.initialize)

# Register routes
app.include_router(main_router)
app.include_router(skill_router)
app.include_router(complevel_router)
app.include_router(metadata_router)

if __name__ == "__main__":
    LOGGING_CONFIG["formatters"]["access"][
        "fmt"
    ] = '%(asctime)s %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
    uvicorn.run(app, host="0.0.0.0", port=8000)
