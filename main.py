from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx
import os
import time
import json
from collections import defaultdict
from datetime import datetime, timezone, timedelta

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://labscoutai.com", "https://www.labscoutai.com", "https://marvelous-hotteok-56e32f.netlify.app"],
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

# Rate limiter: 10 requests per IP per hour
rate_limit_store: dict = defaultdict(list)
RATE_LIMIT = 10
RATE_WINDOW = 3600

def check_rate_limit(ip: str) -> bool:
    now = time.time()
    rate_limit_store[ip] = [t for t in rate_limit_store[ip] if now - t < RATE_WINDOW]
    if len(rate_limit_store[ip]) >= RATE_LIMIT:
        return False
    rate_limit_store[ip].append(now)
    return True

class ScoutRequest(BaseModel):
    paper_text: str

PROMPT_TEMPLATE = """You are a research intelligence analyst. Analyze this AI/ML research paper and extract intelligence for teams working at the intersection of human data and AI research.

PAPER TEXT:
{paper_text}

Return ONLY a JSON object with these exact fields:
{{
  "lab": "Primary institution/lab name (e.g. Stanford HAI, DeepMind, MIT CSAIL, or Unknown)",
  "authors": ["Author 1", "Author 2", "Author 3"],
  "focus": "2-3 sentence summary of what this research is about and what problem it solves",
  "tags": ["tag1", "tag2", "tag3", "tag4"],
  "signal_strength": 85,
  "signal_reason": "2 sentences explaining the signal strength. Does this lab need human-annotated data, participant studies, behavioral research, alignment/RLHF work, or human evaluation?",
  "human_data_need": "1-2 sentences on the specific human data or study infrastructure need this research implies",
  "research": "2-3 sentences on the research context, collaboration potential, and why this work is significant in the broader AI landscape"
}}

Signal strength scoring:
- 90-100: Core RLHF, alignment, human feedback, behavioral annotation, participant studies
- 70-89: LLM/NLP work requiring human eval, preference data, or annotation pipelines
- 50-69: AI/ML research that would benefit from human data or evaluation
- 20-49: Technical AI work with limited direct human data needs
- 0-19: Pure theory, hardware, infrastructure

Return ONLY valid JSON. No markdown, no explanation."""

@app.post("/scout")
async def scout(request: Request, body: ScoutRequest):
    ip = request.client.host

    if not check_rate_limit(ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in an hour.")

    if not body.paper_text or len(body.paper_text.strip()) < 50:
        raise HTTPException(status_code=400, detail="Paper text too short — paste at least an abstract.")

    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="API not configured.")

    prompt = PROMPT_TEMPLATE.format(paper_text=body.paper_text[:8000])

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}],
            },
        )

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail="Upstream error — try again.")

    raw = response.json()["content"][0]["text"].strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        result = json.loads(raw)
    except Exception:
        raise HTTPException(status_code=502, detail="Failed to parse response.")

    return JSONResponse(content=result)

@app.get("/stats")
async def stats():
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise HTTPException(status_code=500, detail="Supabase not configured.")

    week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

    async with httpx.AsyncClient(timeout=15) as client:
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type": "application/json",
        }

        # Total papers this week
        total_resp = await client.get(
            f"{SUPABASE_URL}/rest/v1/items",
            headers={**headers, "Prefer": "count=exact", "Range": "0-0"},
            params={"created_at": f"gte.{week_ago}", "select": "id"},
        )

        # High signal this week (icp_score >= 70)
        high_resp = await client.get(
            f"{SUPABASE_URL}/rest/v1/items",
            headers={**headers, "Prefer": "count=exact", "Range": "0-0"},
            params={"created_at": f"gte.{week_ago}", "icp_score": "gte.70", "select": "id"},
        )

    def parse_count(resp):
        cr = resp.headers.get("content-range", "0/0")
        try:
            return int(cr.split("/")[-1])
        except Exception:
            return 0

    total = parse_count(total_resp)
    high_signal = parse_count(high_resp)

    return JSONResponse(content={
        "papers_this_week": total,
        "high_signal": high_signal,
    })

@app.get("/health")
async def health():
    return {"status": "ok"}
