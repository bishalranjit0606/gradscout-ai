#!/usr/bin/env python3
"""Daily GenAI Master's research agent: find settle-ready programs and email them."""

from __future__ import annotations

import json
import logging
import os
import smtplib
import ssl
import sys
from datetime import date, datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

SEEN_PROGRAMS_PATH = Path(__file__).resolve().parent / "seen_programs.json"
LAST_REPLY_PATH = Path(__file__).resolve().parent / "last_gemini_reply.txt"
MODEL_NAME = "gemini-3.5-flash"
MODEL_FALLBACKS = (
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.1-flash-lite",
)
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465
RECIPIENTS = [
    "bishalranjit2002@gmail.com",
    "bishalranjitofficial@gmail.com",
]
REQUIRED_ENV_VARS = (
    "GEMINI_API_KEY",
    "GMAIL_SENDER_EMAIL",
    "GMAIL_APP_PASSWORD",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gradscout")

SYSTEM_PROMPT = """
You are a practical study-abroad advisor for a regular Nepali student, not
an elite-admissions consultant. Think like a Nepali boy who wants to go
abroad, study, work part-time, pay rent, and stay for years. The goal is
the best university AND city to APPLY to, not a famous name.

Rank in this order:
1. Can I settle here and find a junior IT/AI job with normal effort?
2. Can I get in, afford it, and find a room?
3. Is the course real GenAI / LLM / agent engineering?
University prestige is last.

APPLICANT PROFILE
- Nationality: Nepali passport. Visa must be realistically obtainable.
- Degree: 3-year BCA. GPA 3.29 / 4.00. Average student, not a topper.
- Lifestyle: work + study. Needs legal student part-time work, enough
  student jobs, and rent that a part-time wage can help cover.
- Horizon: long-term. After the Master's he should be able to work in
  that same city/region and stay. A degree in a city with no junior
  hiring is a wasted move.

COURSE FOCUS (strict)
ONLY recommend Master's programs whose core taught content is modern
Generative AI / LLMs / Agentic AI / AI Engineering. Look for names or
tracks like: Generative AI, LLM Engineering, Foundation Models, Agentic AI,
AI Agents, Applied AI Engineering, Multimodal Generative AI, NLP with LLMs,
Prompting + RAG + tool-use, LLMOps / AI systems for serving models.
The curriculum must include several of: LLMs, transformers for generation,
RAG, agents/tools, fine-tuning, diffusion/multimodal gen, AI product
engineering. List 3-6 real module names from the official page.

HARD EXCLUDE COURSES
- Generic MSc Computer Science / Informatics / Software Engineering with
  only an optional "AI track" or one ML elective.
  Exception: TU Graz MSc Computer Science is allowed if the applicant
  takes the Machine Learning or Intelligent Systems major with generative
  / LLM / NLP modules.
- Traditional AI/ML: classical ML, statistics, data mining, "Image
  Processing & AI", generic Applied Informatics.
- Coding-only degrees: DSA, algorithms, compilers, pure software
  development, cloud without GenAI.
- Generic Data Science / Business Analytics.
If you cannot prove GenAI/agent/AI-engineer content from the official
page, skip the program. Do not stretch a generic CS degree to fit.

SETTLE TEST (strict, as important as the course)
Judge the CITY's real junior job market. Do not judge the university brand.

Think Kathmandu vs Pokhara:
- Kathmandu: thousands of IT companies. A junior can keep applying and
  land something. This is the pattern to copy.
- Pokhara: nice city, some good colleges, tourist economy. IT hiring is
  thin. A degree there does not make finding an IT job easy. SKIP this
  pattern: pretty campus town / tourist city with a weak local IT cluster.

Company COUNT vs POPULATION, not logos:
- "Google / IBM / Microsoft / Amazon have an office here" is NOT a reason
  to recommend. Big-tech offices hire few juniors, often locals or seniors.
  Ignore brand-name employers as proof of an easy job market.
- Estimate how many software / IT / AI / product companies actually hire
  in that city or a short commute (local startups, agencies, mid-size
  product firms, outsourcing, public-sector IT). Use job boards, LinkedIn
  junior listings, and local company directories. Write the number or
  range as search-backed, or write "unverified" and skip if you cannot
  show a real cluster.
- Also weigh population and competition. Bengaluru-style is a reject:
  huge company count AND a huge graduate/migrant population, so a first
  junior job is still hard. Mega job magnets where the whole country
  applies are not "easy jobs".
- Prefer a mid-size city with many companies relative to the number of
  people hunting the same junior IT/AI roles. Calm enough to live. Dense
  enough that a Nepali junior can find work without leaving the city.
- If most grads leave after the degree because local hiring is weak, skip.
- If the city is tourism or hospitality first, skip unless search shows a
  real IT hiring cluster, not a handful of companies.

MONEY (scholarship first, settle-fee allowed)
- Prefer scholarships, fee waivers, or low / no tuition.
- If the city PASSES the settle test, a decent self-paid international
  fee is allowed. Decent means a Nepali family can stretch for it because
  staying and working after is realistic.
- Do not recommend luxury / elite pricing unless a scholarship covers
  most of it.
- Never recommend a high fee in a Pokhara-style city (good campus, weak
  jobs). Paying is only worth it when the local job market is real.

EMAIL RULE (strict, most days should be ZERO programs)
This job runs daily. Do NOT send a daily digest. Do NOT pad a list of
3-4 universities. Count can be 0, 1, or many. Zero is the normal result.
Include a program ONLY if ALL of these are true:
- It matches every course, settle, money, and place rule in this prompt.
- Applications are actually open now (Apply is live), or the official
  window opens within the next 14 days.
- The official deadline is today or in the future (verified date).
- It is not already in the seen list for this university+course+intake.
SKIP closed rounds, guessed dates, and "opens next month" beyond 14 days.
If nothing new and settle-ready is open, return
new_opportunities_found=false and empty lists.
Do not reuse a university+course already in the seen list unless this is
a NEW intake (for example autumn-2028 after autumn-2027 was already sent).

WORKING LINKS ONLY (strict)
- Use Google Search. Open/read official university pages, not blogs or
  aggregators (no educations.com / mastersportal / shiksha as the main link).
- Every Official Link must be a full https URL to the program page or the
  official apply page that currently exists.
- Do not use truncated hosts (admissions.ktu.edu with no path), homepages,
  or guessed URLs. If search does not show a live program URL, skip it.
- Prefer links that show Apply / How to apply / tuition / modules.

HARD EXCLUDE PLACES
- Countries: USA, Canada, Australia, United Kingdom, Germany.
- Overcrowded / expensive magnets: Amsterdam, Dublin, Paris, Barcelona,
  Madrid, Rome, Milan, Zurich, Geneva, Singapore, Hong Kong, Seoul, Tokyo,
  Osaka, Stockholm. Helsinki city-center if rent is brutal.
- Bengaluru-style mega IT cities: huge company count, but population and
  applicant volume make a junior role a fight.
- Prestige-first picks unless the city is calm AND admission + living costs
  are realistic for this GPA AND the settle test passes.

GROUND REALITY (strict, not brochure / not "on paper")
Judge the CITY as a Nepali student would live it, not the country's marketing.
- Safety: skip places with a real reputation for street crime, knife/gun
  violence, unsafe nights, scams on internationals, or students saying they
  do not walk home after dark. Official "safe country" rankings are not
  enough. USA-style paper-safe / real-unsafe is exactly what to avoid.
  Prefer genuinely calm student towns where people actually feel safe.
- After Master's job: only recommend if a non-EU / Nepali graduate can
  find junior AI / LLM / software work in THAT city/region with normal
  effort, because many companies hire juniors there, not because one
  famous company has an office. Say typical starting pay students report
  (2025-2026), not the university's "average salary" poster. Skip if
  grads mostly leave, or pay cannot cover rent.
- Tuition: use the real international fee students pay this cycle, plus
  extra costs (application, residence permit, health insurance, nostrification).
  Say if it is scholarship, low-fee, or a decent self-pay that is worth
  it only because the city is settle-ready.
- Rent: use what students actually pay now (dorm waitlists, Facebook/housing
  groups, recent posts). Brochure "from €120" is useless if dorms are full
  and private rooms are double that.
- Part-time jobs: legal hours are not enough. Say whether Nepali/English
  speakers actually get campus, warehouse, delivery, cafe, or junior IT
  shifts in that city, and a realistic hourly wage.
- If search shows mixed/bad ground reports on safety or jobs, skip or be
  blunt in downsides. Do not polish a risky city to sound nice.

PRIORITY (search and rank in this order)
Only keep a priority university if it also passes the settle test.
1. FIRST, these exact universities/courses. Watch their official apply
   pages every run. Include them if the intake is open (or opens in 14
   days) and the city still has a real junior IT/AI cluster:
   - FH JOANNEUM, Graz: Machine Learning and Generative AI
     https://www.fh-joanneum.at/machine-learning-and-generative-ai/master/en/
   - TU Graz: MSc Computer Science, Machine Learning or Intelligent Systems
     major (Generative Deep Learning / NLP). This CS wrapper is ALLOWED
     only with that AI major, because the applicant chose Graz.
     https://www.tugraz.at/en/studying-and-teaching/degree-and-certificate-programmes/masters-degree-programmes/computer-science
     Apply window for 2027/28: 15 October to 15 December 2026.
   - JKU Linz: MSc Artificial Intelligence (LLMs / generative methods)
     https://www.jku.at/en/degree-programs/types-of-degree-programs/masters-degree-programs/ma-artificial-intelligence
     Non-EU winter intake: typically 6 February to 31 March.
2. SECOND, other Austria GenAI / LLM / AI-engineering Master's in calm
   cities with a real local IT cluster: Graz, Linz, Innsbruck, Klagenfurt.
   Skip Vienna (too crowded) unless nothing else is open and it still
   passes every other filter. Prefer four-season climate (not extreme
   Nordic cold). English-taught is OK; note that jobs and PR in Austria
   need German.
3. THIRD, other EU cities that pass the settle test better than Austria
   if Austria's junior IT cluster is thin: same exclude list, same
   GenAI course rule, applications open now. Do not pad.

Do not fill the email with random EU schools. Only the best apply-now
fits that pass course + settle + money.

PREFER THESE KINDS OF PLACES
- Peaceful mid-size cities with a Kathmandu-style IT cluster: many local
  companies, not a tourist-first town, not a mega-city crush.
- Graz-style living is still the dream IF junior IT/AI hiring is real
  there: safe, not too hot, not Arctic-cold, house later in suburbs.
- After Austria, same style in EU only if the job-density test passes:
  Braga/Coimbra (Portugal), Ljubljana, Brno, Padova/Trento, smaller
  France (Toulouse, Grenoble, Nantes). Skip any of these if they are
  Pokhara-style (nice, weak IT hiring).
- Average / applied universities (FH) are welcome. Elite branding is not.

ADMISSION REALITY FILTER
- 3-year South Asian / Nepali bachelor's accepted, or a real pre-master.
- GPA bar this applicant can clear. English-taught. IELTS 6.0-6.5 preferred.

FOR EACH PROGRAM INCLUDE
- University, city, country, exact program + track name
- 3-6 official module names proving GenAI / agents / AI engineering
- Full working https official URL
- Application status: opened on [date], deadline, intake term (verified)
- Why this city is settle-ready: company-count range, population /
  competition note, Kathmandu-style vs Pokhara-style
- City vibe, REAL student-room rent (not brochure), part-time job reality
  (hours + whether jobs actually exist + typical hourly pay)
- Real safety note (night walking, student reports, not a ranking table)
- After-degree junior hiring in THAT city and typical junior pay
- Tuition + hidden extras: scholarship / low-fee / decent self-pay and
  why the fee is worth it (or not)
- 3-year BCA eligibility, language test
- Why it is achievable; honest downsides

SEARCH RULES
- Ignore previously-seen program IDs.
- If a fact is unsure, write "unverified" or skip the program.
- Never invent deadlines, rent, company counts, or URLs.
- Never use "big companies exist here" as the job argument.
- Rank by: (1) settle test (company count vs population, junior hiring),
  (2) applications open now or within 14 days, (3) GenAI/agent curriculum,
  (4) working official URL, (5) real safety / rent / part-time work,
  (6) scholarship or decent settle-worthy fee, (7) visa practicality,
  (8) Austria watch-list only if it still passes settle.

PROGRAM ID FORMAT
- lowercase: country|university-slug|program-slug|intake-term-year
  Example: "at|fh-joanneum-graz|ml-generative-ai|winter-2027"
- course identity is the first three parts (university + course). The last
  part is the intake. Same course + same intake must never be emailed twice.
  Same course is allowed again only for a later intake.

OUTPUT CONTRACT
Return ONLY valid JSON (no markdown fences, no commentary):
{
  "new_opportunities_found": boolean,
  "discovered_programs": [
    {
      "id": "at|fh-joanneum-graz|ml-generative-ai|winter-2027",
      "application_opened_on": "YYYY-MM-DD",
      "application_deadline": "YYYY-MM-DD"
    }
  ],
  "html_report": string
}

- new_opportunities_found: true only if at least one program is open
  (or opens within 14 days), passes every filter including the settle
  test, and is not already in the seen list. Otherwise false.
- application_opened_on: official apply-start date if known.
- application_deadline: official deadline (must be today or later).
  If you cannot verify the deadline, omit the program.
- html_report: email-safe HTML (inline CSS). Practical work/study/live tone.
  Each card must show: modules, apply window + deadline, full clickable
  https link, company-count vs population, why it is not a tourist-campus
  trap, real rent, real part-time work, real safety, after-master junior
  hiring, fee type (scholarship / low / decent self-pay). Dark-on-light,
  mobile readable.

If nothing new and settle-ready is open, set new_opportunities_found to
false, discovered_programs to [], and html_report to "".
Better to return zero programs than a closed, tourist-city, mega-city,
or generic CS program.
""".strip()


def load_local_env(path: Path | None = None) -> None:
    """Load KEY=VALUE pairs from a local .env file without extra dependencies."""
    env_path = path or Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def require_env() -> None:
    missing = [name for name in REQUIRED_ENV_VARS if not os.environ.get(name)]
    if missing:
        raise EnvironmentError(
            "Missing required environment variables: " + ", ".join(missing)
        )


def parse_program_id(program_id: str) -> tuple[str, str]:
    """Split a program id into course_key (uni+course) and intake."""
    parts = [part.strip().lower() for part in program_id.split("|") if part.strip()]
    if len(parts) >= 4:
        return "|".join(parts[:3]), parts[-1]
    if len(parts) == 3:
        return "|".join(parts), ""
    return program_id.strip().lower(), ""


def load_seen_programs(path: Path = SEEN_PROGRAMS_PATH) -> list[dict[str, str]]:
    if not path.exists():
        logger.info("No seen-programs file found. Initializing empty list.")
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse %s (%s). Starting from an empty list.", path, exc)
        return []

    if not isinstance(data, list):
        logger.warning("%s is not a JSON array. Starting from an empty list.", path)
        return []

    seen: list[dict[str, str]] = []
    for item in data:
        if isinstance(item, str) and item.strip():
            course_key, intake = parse_program_id(item)
            seen.append({"course_key": course_key, "intake": intake, "id": item.strip()})
        elif isinstance(item, dict):
            course_key = str(item.get("course_key") or "").strip().lower()
            intake = str(item.get("intake") or "").strip().lower()
            full_id = str(item.get("id") or "").strip().lower()
            if not course_key and full_id:
                course_key, parsed_intake = parse_program_id(full_id)
                intake = intake or parsed_intake
            if course_key:
                seen.append(
                    {
                        "course_key": course_key,
                        "intake": intake,
                        "id": full_id or f"{course_key}|{intake}".rstrip("|"),
                    }
                )
    return seen


def save_seen_programs(records: list[dict[str, str]], path: Path = SEEN_PROGRAMS_PATH) -> None:
    unique: list[dict[str, str]] = []
    index: dict[tuple[str, str], int] = {}
    for record in records:
        key = (record["course_key"], record.get("intake") or "")
        if key in index:
            unique[index[key]] = record
            continue
        index[key] = len(unique)
        unique.append(record)
    path.write_text(json.dumps(unique, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved %d seen course record(s) to %s.", len(unique), path)


def already_emailed(seen: list[dict[str, str]], course_key: str, intake: str) -> bool:
    for record in seen:
        if record["course_key"] != course_key:
            continue
        stored_intake = record.get("intake") or ""
        if not intake or not stored_intake or stored_intake == intake:
            return True
    return False


def parse_iso_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None


def opened_today_or_yesterday(opened_on: str, today: date) -> bool:
    opened = parse_iso_date(opened_on)
    if opened is None:
        return False
    return opened in {today, today - timedelta(days=1)}


def applications_still_open(opened_on: str, deadline: str, today: date) -> bool:
    due = parse_iso_date(deadline)
    if due is not None:
        return due >= today
    return opened_today_or_yesterday(opened_on, today)


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output, including fenced or noisy text."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```JSON").removeprefix("```")
        cleaned = cleaned.removesuffix("```").strip()

    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    index = 0
    while index < len(cleaned):
        brace = cleaned.find("{", index)
        if brace == -1:
            break
        try:
            parsed, offset = decoder.raw_decode(cleaned[brace:])
        except json.JSONDecodeError:
            index = brace + 1
            continue
        if isinstance(parsed, dict):
            candidates.append(parsed)
        index = brace + max(offset, 1)

    if not candidates:
        snippet = cleaned[:200].replace("\n", " ")
        raise ValueError(f"Model response did not contain a JSON object. Start: {snippet!r}")

    for candidate in candidates:
        if "new_opportunities_found" in candidate and (
            "discovered_programs" in candidate
            or "discovered_program_ids" in candidate
            or "html_report" in candidate
        ):
            return candidate
    return candidates[0]


def build_user_prompt(seen_records: list[dict[str, str]]) -> str:
    today = date.today().isoformat()
    seen_json = json.dumps(seen_records, indent=2)
    return (
        f"Today's date: {today}. Treat this date as ground truth.\n\n"
        "This is a daily check for the BEST university + city to apply to, "
        "not a daily newsletter. Most days return ZERO programs. Only include "
        "a course if applications are open now (or open within 14 days), the "
        "deadline is today or later, the course is real GenAI, AND the city "
        "passes the settle test. Do not pad 3-4 universities. No limit: 0 to many.\n\n"
        "SETTLE TEST: copy Kathmandu (many IT companies, junior can land a "
        "job), reject Pokhara (nice colleges, tourist city, thin IT hiring). "
        "Use company COUNT vs POPULATION. Do not recommend because Google, "
        "IBM, or another big brand has an office. Reject Bengaluru-style "
        "mega cities where company count is high but competition is brutal.\n\n"
        "MONEY: prefer scholarship or low fee. A decent self-paid fee is OK "
        "only if the city is settle-ready.\n\n"
        "Check FIRST: FH JOANNEUM Graz (Machine Learning and Generative AI), "
        "TU Graz Computer Science (ML / Intelligent Systems major), and "
        "JKU Linz MSc Artificial Intelligence, but only if they still pass "
        "the settle test. SECOND: other Austria in Graz, Linz, Innsbruck, "
        "Klagenfurt (skip Vienna). THIRD: other EU cities that pass settle "
        "better. Do not pad with random universities.\n\n"
        "Find Master's programs focused on Generative AI, LLMs, AI agents, or "
        "AI engineering (not generic CS, not classical ML, not coding-only, "
        "except the TU Graz CS+ML major above). "
        "Every Official Link must be a full https URL that search shows as a "
        "live university page. Peaceful mid-size cities that are actually "
        "safe and have a real junior IT/AI cluster. Use ground-reality rent, "
        "tuition, company counts, population, part-time jobs, and after-master "
        "hiring, not brochure numbers. Skip USA, Canada, Australia, UK, and "
        "Germany. Skip elite schools and tourist-campus towns.\n\n"
        "ALREADY EMAILED (do not send the same university+course again unless "
        "the intake is new):\n"
        f"{seen_json}\n\n"
        "Return JSON only, using the required schema."
    )


def _error_code(exc: BaseException) -> int | None:
    if isinstance(exc, genai_errors.ClientError):
        return exc.code
    return None


def _generate(
    client: genai.Client,
    model: str,
    contents: str,
    *,
    json_mode: bool,
    use_search: bool,
) -> Any:
    config_kwargs: dict[str, Any] = {
        "system_instruction": SYSTEM_PROMPT,
        "temperature": 0.2,
    }
    if use_search:
        config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"
    return client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(**config_kwargs),
    )


def _generate_with_json_fallback(
    client: genai.Client,
    model: str,
    contents: str,
    *,
    use_search: bool,
) -> Any:
    try:
        return _generate(client, model, contents, json_mode=True, use_search=use_search)
    except Exception as json_mode_error:
        code = _error_code(json_mode_error)
        if code in {404, 429}:
            raise
        logger.warning(
            "JSON mime type failed on %s (%s). Retrying without mime type.",
            model,
            json_mode_error,
        )
        return _generate(client, model, contents, json_mode=False, use_search=use_search)


def research_programs(client: genai.Client, seen_records: list[dict[str, str]]) -> dict[str, Any]:
    contents = build_user_prompt(seen_records)
    models_to_try = (MODEL_NAME,) + tuple(
        name for name in MODEL_FALLBACKS if name != MODEL_NAME
    )
    last_error: Exception | None = None
    response = None
    quota_hit = False

    for model in models_to_try:
        try:
            logger.info("Querying %s with Google Search grounding...", model)
            response = _generate_with_json_fallback(
                client, model, contents, use_search=True
            )
            break
        except Exception as model_error:
            last_error = model_error
            code = _error_code(model_error)
            if code == 429:
                quota_hit = True
                logger.warning(
                    "Quota exceeded on %s (often Google Search grounding on the free plan). "
                    "Retrying once without live web search.",
                    model,
                )
                break
            logger.warning("Model %s failed (%s). Trying the next candidate.", model, model_error)
    else:
        raise RuntimeError("All Gemini model candidates failed.") from last_error

    if response is None:
        logger.info("Querying %s without Google Search (model knowledge only)...", MODEL_NAME)
        try:
            response = _generate_with_json_fallback(
                client, MODEL_NAME, contents, use_search=False
            )
        except Exception as fallback_error:
            if quota_hit:
                raise RuntimeError(
                    "Gemini quota is exhausted (often the Google Search grounding limit "
                    "on the free plan). Wait a few minutes, check "
                    "https://ai.dev/rate-limit, or enable billing in Google AI Studio."
                ) from fallback_error
            raise RuntimeError("Gemini research call failed.") from fallback_error

    raw_text = (response.text or "").strip()
    LAST_REPLY_PATH.write_text(raw_text + "\n", encoding="utf-8")
    logger.info("Saved full Gemini reply to %s (%d chars).", LAST_REPLY_PATH, len(raw_text))
    preview = raw_text if len(raw_text) <= 1200 else raw_text[:1200] + "\n...[truncated]"
    logger.info("Gemini reply preview:\n%s", preview)
    if not raw_text:
        raise RuntimeError("Gemini returned an empty response.")

    payload = extract_json_object(raw_text)
    if "new_opportunities_found" not in payload:
        raise ValueError("Model JSON is missing 'new_opportunities_found'.")
    if "html_report" not in payload:
        raise ValueError("Model JSON is missing 'html_report'.")
    if "discovered_programs" not in payload and "discovered_program_ids" not in payload:
        raise ValueError("Model JSON is missing 'discovered_programs'.")
    return payload


def _iter_discovered(payload: dict[str, Any]) -> list[dict[str, str]]:
    programs = payload.get("discovered_programs")
    parsed: list[dict[str, str]] = []
    if isinstance(programs, list):
        for item in programs:
            if isinstance(item, str) and item.strip():
                parsed.append(
                    {
                        "id": item.strip(),
                        "application_opened_on": "",
                        "application_deadline": "",
                    }
                )
            elif isinstance(item, dict):
                program_id = str(item.get("id") or "").strip()
                opened_on = str(item.get("application_opened_on") or "").strip()
                deadline = str(item.get("application_deadline") or "").strip()
                if program_id:
                    parsed.append(
                        {
                            "id": program_id,
                            "application_opened_on": opened_on,
                            "application_deadline": deadline,
                        }
                    )
        return parsed

    raw_ids = payload.get("discovered_program_ids") or []
    if isinstance(raw_ids, list):
        for item in raw_ids:
            if isinstance(item, str) and item.strip():
                parsed.append(
                    {
                        "id": item.strip(),
                        "application_opened_on": "",
                        "application_deadline": "",
                    }
                )
    return parsed


def normalize_result(
    payload: dict[str, Any],
    seen_records: list[dict[str, str]],
    today: date | None = None,
) -> dict[str, Any]:
    today = today or date.today()
    discovered: list[dict[str, str]] = []
    seen_in_batch: set[tuple[str, str]] = set()

    for item in _iter_discovered(payload):
        program_id = item["id"]
        opened_on = item.get("application_opened_on") or ""
        deadline = item.get("application_deadline") or ""
        if not applications_still_open(opened_on, deadline, today):
            logger.info(
                "Skipping %s: deadline=%r opened_on=%r is not an open window.",
                program_id,
                deadline or "missing",
                opened_on or "missing",
            )
            continue

        course_key, intake = parse_program_id(program_id)
        if already_emailed(seen_records, course_key, intake):
            logger.info(
                "Skipping %s: already emailed this university+course for this intake.",
                program_id,
            )
            continue

        batch_key = (course_key, intake)
        if batch_key in seen_in_batch:
            continue
        seen_in_batch.add(batch_key)
        discovered.append(
            {
                "id": program_id,
                "course_key": course_key,
                "intake": intake,
                "application_opened_on": opened_on,
                "application_deadline": deadline,
            }
        )

    html_report = payload.get("html_report") or ""
    if not isinstance(html_report, str):
        html_report = str(html_report)

    found = bool(discovered)
    if found and not html_report.strip():
        logger.warning("Matching courses found but html_report is empty. Skipping email.")
        found = False

    return {
        "new_opportunities_found": found,
        "discovered_programs": discovered,
        "html_report": html_report,
    }


def send_html_email(html_body: str, program_count: int) -> None:
    sender = os.environ["GMAIL_SENDER_EMAIL"].strip()
    app_password = os.environ["GMAIL_APP_PASSWORD"].replace(" ", "")
    subject = (
        f"GradScout: settle-ready apply options "
        f"({program_count} matching course{'s' if program_count != 1 else ''})"
    )

    message = MIMEMultipart("alternative")
    message["From"] = sender
    message["To"] = ", ".join(RECIPIENTS)
    message["Subject"] = subject
    message["Date"] = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")
    message.attach(MIMEText(html_body, "html", "utf-8"))

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context, timeout=60) as server:
        server.login(sender, app_password)
        server.sendmail(sender, RECIPIENTS, message.as_string())

    logger.info("HTML email sent to %s.", ", ".join(RECIPIENTS))


def main() -> int:
    load_local_env()
    require_env()

    seen_records = load_seen_programs()
    logger.info("Loaded %d previously emailed course record(s).", len(seen_records))

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    try:
        payload = research_programs(client, seen_records)
    except Exception:
        logger.exception("Gemini research call failed.")
        return 1

    result = normalize_result(payload, seen_records)
    new_records = result["discovered_programs"]

    if not result["new_opportunities_found"]:
        logger.info(
            "No new settle-ready courses are open. Exiting without sending email."
        )
        return 0

    new_ids = [record["id"] for record in new_records]
    logger.info("Settle-ready apply options for %d course(s): %s", len(new_ids), ", ".join(new_ids))
    save_seen_programs(seen_records + new_records)

    try:
        send_html_email(result["html_report"], len(new_ids))
    except Exception:
        logger.exception("Failed to send email. Seen-programs file was already updated.")
        return 1

    logger.info("Daily research run completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
