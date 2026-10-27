"""
HTML page shells. Templates get only {request, page, version}; every number
comes from /api through fetch().

A page whose template (web/templates/<page>.html) doesn't exist yet renders
coming_soon.html, so a milestone ships its page just by adding the template.
"""
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from server import __version__, paths

router = APIRouter(include_in_schema=False)
ORDER = 1000

# page key → URL path (order = nav order, mirrored in base.html)
PAGES = {
    "slate":    "/",
    "game":     "/game",
    "ratings":  "/ratings",
    "accuracy": "/accuracy",
    "bets":     "/bets",
    "model":    "/model",
    "settings": "/settings",
}


def template_for(page: str) -> str:
    name = f"{page}.html"
    return name if (paths.TEMPLATES_DIR / name).exists() else "coming_soon.html"


def _page_route(page: str):
    def render(request: Request) -> HTMLResponse:
        templates = request.app.state.templates
        return templates.TemplateResponse(
            request, template_for(page), {"page": page, "version": __version__})
    render.__name__ = f"page_{page}"
    return render


for _page, _path in PAGES.items():
    router.add_api_route(_path, _page_route(_page), methods=["GET"],
                         response_class=HTMLResponse)


@router.get("/favicon.ico")
def favicon():
    return RedirectResponse("/static/img/favicon.svg", status_code=308)
