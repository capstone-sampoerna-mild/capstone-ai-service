from fastapi import FastAPI
from fastapi_swagger_ui_theme import setup_swagger_ui_theme
from dotenv import load_dotenv
from app.v1.router import router as v1_router

load_dotenv()
app = FastAPI(docs_url=None)

setup_swagger_ui_theme(
    app,
    docs_path="/docs",
    title="API Docs",
    static_mount_path="/swagger-ui-theme-static",
    swagger_favicon_url=None,
    oauth2_redirect_url=None,
    init_oauth=None,
    swagger_ui_parameters=None,
)

app.include_router(v1_router)