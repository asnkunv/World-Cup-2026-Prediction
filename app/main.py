import pandas as pd
from src.simulation import run_monte_carlo, qualified_with_stats, model, scaler, display_results
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pathlib import Path

app = FastAPI()
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

GROUPS = {
    'A': ['brazil', 'denmark', 'egypt', 'uruguay'],
    'B': ['argentina', 'italy', 'morocco', 'panama'],
    'C': ['france', 'croatia', 'senegal', 'germany'],
    'D': ['spain', 'mexico', 'south korea', 'canada'],
    'E': ['england', 'poland', 'japan', 'qatar'],
    'F': ['belgium', 'colombia', 'ghana', 'jordan'],
    'G': ['netherlands', 'united states', 'austria', 'cape verde'],
    'H': ['portugal', 'turkey', 'south africa', 'uzbekistan']
}


@app.get("/", response_class=HTMLResponse)
async def simulation(request: Request):
    results = run_monte_carlo(GROUPS, qualified_with_stats, model, n_simulations=100, scaler=scaler)

    # sort before passing to template
    sorted_champion = sorted(results['champion'].items(), key=lambda x: x[1], reverse=True)

    return templates.TemplateResponse("simulation.html", {
        "request": request,
        "results": results,
        "sorted_teams": sorted_champion
    })
