#!/usr/bin/env python3
"""Daily GenAI Master's research agent: search, dedupe, and email new programs."""

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
abroad, study, work part-time, pay rent, and live a peaceful life. Rank
"can I actually get in, afford it, find a room, and find a job" above
university prestige.

APPLICANT PROFILE
- Nationality: Nepali passport. Visa must be realistically obtainable.
- Degree: 3-year BCA. GPA 3.29 / 4.00. Average student, not a topper.
- Lifestyle: work + study. Needs legal student part-time work, enough
  student jobs, and rent that a part-time wage can help cover.

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

EMAIL RULE (strict, most days should be ZERO programs)
This job runs daily. Do NOT send a daily digest. Do NOT pad a list of
3-4 universities. Count can be 0, 1, or many. Zero is the normal result.
Include a program ONLY if ALL of these are true:
- It matches every requirement in this prompt.
- The official page shows the application window STARTS today (the apply
  portal opened today). Official start date must equal today's date, or
  yesterday if the page says it opened in the last 24 hours (timezone).
- Applications are actually open (Apply is live, not greyed out).
SKIP programs that have already been open for days or weeks.
SKIP closed rounds, guessed future dates, and "opens next month".
If nothing opened today, return new_opportunities_found=false and empty lists.
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
- Prestige-first picks unless the city is calm AND admission + living costs
  are realistic for this GPA.

GROUND REALITY (strict, not brochure / not "on paper")
Judge the CITY as a Nepali student would live it, not the country's marketing.
- Safety: skip places with a real reputation for street crime, knife/gun
  violence, unsafe nights, scams on internationals, or students saying they
  do not walk home after dark. Official "safe country" rankings are not
  enough. USA-style paper-safe / real-unsafe is exactly what to avoid.
  Prefer genuinely calm student towns where people actually feel safe.
- After Master's income: only recommend if a non-EU / Nepali graduate can
  realistically find junior AI / LLM / software work or a clear post-study
  job path in that city/region. Say typical starting pay students report
  (2025-2026), not the university's "average salary" poster. Skip if grads
  mostly leave because there are no jobs, or pay cannot cover rent.
- Tuition: use the real international fee students pay this cycle, plus
  extra costs (application, residence permit, health insurance, nostrification).
- Rent: use what students actually pay now (dorm waitlists, Facebook/housing
  groups, recent posts). Brochure "from €120" is useless if dorms are full
  and private rooms are double that.
- Part-time jobs: legal hours are not enough. Say whether Nepali/English
  speakers actually get campus, warehouse, delivery, cafe, or junior IT
  shifts in that city, and a realistic hourly wage.
- If search shows mixed/bad ground reports on safety or jobs, skip or be
  blunt in downsides. Do not polish a risky city to sound nice.

PRIORITY (search and rank in this order)
1. FIRST, these exact universities/courses (maximum focus). Watch their
   official apply pages every run. Email if a matching intake opens today:
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
   cities: Graz, Linz, Innsbruck, Klagenfurt. Skip Vienna (too crowded)
   unless nothing else opened and it still matches every other filter.
   Prefer four-season climate (not extreme Nordic cold). English-taught
   is OK; note that jobs and PR in Austria need German.
3. THIRD, only if nothing in (1) or (2) opened today: other EU countries
   that still pass every filter (GenAI course, open today, safe, rent,
   part-time work, not overcrowded). Same exclude list still applies.

Do not pad. If the priority universities did not open today, returning
zero programs is correct. Do not fill the email with random EU schools.

PREFER THESE KINDS OF PLACES
- Peaceful mid-size student cities near nature. Graz-style living is the
  dream: safe, not too hot, not Arctic-cold, house later in suburbs.
- After Austria, same style in EU: Braga/Coimbra (Portugal), Ljubljana,
  Brno, Padova/Trento, smaller France (Toulouse, Grenoble, Nantes).
- Average / applied universities (FH) are welcome. Elite branding is not.

ADMISSION REALITY FILTER
- 3-year South Asian / Nepali bachelor's accepted, or a real pre-master.
- GPA bar this applicant can clear. English-taught. IELTS 6.0-6.5 preferred.

FOR EACH PROGRAM INCLUDE
- University, city, country, exact program + track name
- 3-6 official module names proving GenAI / agents / AI engineering
- Full working https official URL
- Application status: opened on [date], deadline, intake term (verified)
- City vibe, REAL student-room rent (not brochure), part-time job reality
  (hours + whether jobs actually exist + typical hourly pay)
- Real safety note (night walking, student reports, not a ranking table)
- After-degree job reality and typical junior pay for internationals
- Tuition + hidden extras, realistic scholarships
- 3-year BCA eligibility, language test
- Why it is achievable; honest downsides

SEARCH RULES
- Ignore previously-seen program IDs.
- If a fact is unsure, write "unverified" or skip the program.
- Never invent deadlines, rent, or URLs.
- Rank by: (1) application actually opened today, (2) priority list
  (FH JOANNEUM Graz, then TU Graz, then JKU Linz, then other Austria,
  then other EU), (3) GenAI/agent/AI-engineer curriculum, (4) working
  official URL, (5) real safety / rent / jobs, (6) visa practicality.

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
      "application_opened_on": "YYYY-MM-DD"
    }
  ],
  "html_report": string
}

- new_opportunities_found: true only if at least one program opened today
  AND passes every filter. Otherwise false.
- application_opened_on: the official apply-start date (must be today or
  yesterday). If you cannot verify that date, omit the program.
- html_report: email-safe HTML (inline CSS). Practical work/study/live tone.
  Each card must show: modules, "applications opened today", deadline,
  full clickable https link, real rent, real part-time work, real safety,
  after-master pay. Dark-on-light, mobile readable.

If nothing opened today, set new_opportunities_found to false,
discovered_programs to [], and html_report to "".
Better to return zero programs than a closed, old, or generic CS program.
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


def opened_today_or_yesterday(opened_on: str, today: date) -> bool:
    try:
        opened = date.fromisoformat(opened_on.strip())
    except ValueError:
        return False
    return opened in {today, today - timedelta(days=1)}


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
        "This is a daily check, not a daily newsletter. Most days return "
        "ZERO programs. Only include a course if its official application "
        "window opened TODAY (or yesterday, timezone catch-up) AND it matches "
        "every requirement. Do not pad 3-4 universities. No limit: 0 to many.\n\n"
        "Check FIRST: FH JOANNEUM Graz (Machine Learning and Generative AI), "
        "TU Graz Computer Science (ML / Intelligent Systems major), and "
        "JKU Linz MSc Artificial Intelligence. SECOND: other Austria in Graz, "
        "Linz, Innsbruck, Klagenfurt (skip Vienna). THIRD: other EU only if "
        "nothing in Austria opened today. Do not pad with random universities.\n\n"
        "Find Master's programs focused on Generative AI, LLMs, AI agents, or "
        "AI engineering (not generic CS, not classical ML, not coding-only, "
        "except the TU Graz CS+ML major above). "
        "Every Official Link must be a full https URL that search shows as a "
        "live university page. Peaceful smaller cities that are actually safe "
        "to live in, not only safe on paper. Use ground-reality rent, tuition, "
        "part-time jobs, and after-master income, not brochure numbers. Skip "
        "USA, Canada, Australia, UK, and Germany. Skip elite schools.\n\n"
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
                parsed.append({"id": item.strip(), "application_opened_on": ""})
            elif isinstance(item, dict):
                program_id = str(item.get("id") or "").strip()
                opened_on = str(item.get("application_opened_on") or "").strip()
                if program_id:
                    parsed.append({"id": program_id, "application_opened_on": opened_on})
        return parsed

    raw_ids = payload.get("discovered_program_ids") or []
    if isinstance(raw_ids, list):
        for item in raw_ids:
            if isinstance(item, str) and item.strip():
                parsed.append({"id": item.strip(), "application_opened_on": ""})
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
        if not opened_today_or_yesterday(opened_on, today):
            logger.info(
                "Skipping %s: application_opened_on=%r is not today/yesterday.",
                program_id,
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
        f"GradScout: admissions opened today "
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
            "No matching courses opened today. Exiting without sending email."
        )
        return 0

    new_ids = [record["id"] for record in new_records]
    logger.info("Admissions opened today for %d course(s): %s", len(new_ids), ", ".join(new_ids))
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
