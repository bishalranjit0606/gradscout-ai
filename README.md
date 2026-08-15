# Daily GenAI Master's Research & Email Automation Agent

GradScout is a daily research agent. It searches the live web with Gemini 2.5 Flash (Google Search grounding), finds new Master's programs in Generative AI, LLMs, and Agentic AI, and emails an HTML briefing when something new appears.

The target profile is a **Nepali applicant with a BCA (GPA 3.29)**. Traditional ML, statistics, and generic data-science degrees are ignored unless the program is clearly GenAI / LLM / agentic.

## What it does

1. Runs every day, but most days sends **no email**.
2. Emails only if a matching GenAI / AI-agent course **opened applications today**.
3. No daily quota of 3-4 universities. Count can be 0, 1, or many.
4. The same university + course is emailed once per intake. It can be sent again only for the next intake.

Emails go to `bishalranjit2002@gmail.com` and `bishalranjitofficial@gmail.com`.

## Directory structure

```text
.
├── .github/workflows/daily-research.yml  # Cron + manual GitHub Actions run
├── .gitignore
├── README.md
├── main.py                               # Research + email agent
├── requirements.txt
└── seen_programs.json                    # Dedup store (empty array at start)
```

## Local setup

You need Python 3.11+, a Gemini API key, and a Gmail account with an App Password.

```bash
python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Set these environment variables (or put them in a local `.env` file; `main.py` loads it if present):

```bash
export GEMINI_API_KEY="your-gemini-api-key"
export GMAIL_SENDER_EMAIL="your-gmail-address@gmail.com"
export GMAIL_APP_PASSWORD="your-16-char-app-password"
```

Example `.env`:

```env
GEMINI_API_KEY=your-gemini-api-key
GMAIL_SENDER_EMAIL=your-gmail-address@gmail.com
GMAIL_APP_PASSWORD=your-16-char-app-password
```

Run once:

```bash
python main.py
```

Expected logs:

- `No new Master's opportunities found...` if Gemini finds nothing new.
- `HTML email sent...` if new programs were found.

## GitHub Actions (daily at 06:00 UTC)

The workflow `.github/workflows/daily-research.yml` runs:

- every day at `0 6 * * *` (06:00 UTC)
- on demand via **Actions → Daily GenAI Master's Research → Run workflow**

`seen_programs.json` is persisted between runs with `actions/cache@v4`:

- save key: `program-cache-${{ github.run_id }}`
- restore prefix: `program-cache-`

The repo copy of `seen_programs.json` stays `[]`. The cache is the live memory of already-sent program IDs.

## Configure GitHub Repository Secrets

Create three secrets on the GitHub repo before the first scheduled run.

### 1. `GEMINI_API_KEY`

1. Open [Google AI Studio](https://aistudio.google.com/apikey).
2. Create an API key for a Google Cloud project that has the Gemini API enabled.
3. Copy the key.

### 2. `GMAIL_SENDER_EMAIL`

Use the full Gmail address that will send the briefing (the same account that owns the App Password).

### 3. `GMAIL_APP_PASSWORD`

Gmail SMTP (`smtp.gmail.com:465`, SSL) needs an App Password, not your normal login password.

1. Turn on [2-Step Verification](https://myaccount.google.com/signinoptions/two-step-verification) on that Google account.
2. Open [App passwords](https://myaccount.google.com/apppasswords).
3. Create a password for Mail / Other (`GradScout`).
4. Copy the 16-character password (spaces are fine; the script strips them).

### Add the secrets in GitHub

1. Open the repository on GitHub.
2. Go to **Settings → Secrets and variables → Actions**.
3. Click **New repository secret**.
4. Add each of these names and values:
   - `GEMINI_API_KEY`
   - `GMAIL_SENDER_EMAIL`
   - `GMAIL_APP_PASSWORD`
5. Run **Actions → Daily GenAI Master's Research → Run workflow** once to confirm:
   - Python 3.11 installs
   - cache restore/save works
   - the job either emails new programs or logs that none were found

If the job fails on SMTP, the usual cause is a normal Gmail password instead of an App Password, or 2-Step Verification still off.

## Notes

- The model is `gemini-3.5-flash` (with newer Flash fallbacks) plus Google Search and JSON output.
- Program IDs look like `country|university-slug|program-slug|intake-term-year`.
- If Gemini JSON mode plus Search is rejected by the API, `main.py` retries without the JSON mime type and still parses the JSON object.
- Email send happens only when `new_opportunities_found` is true and there is at least one new program ID plus a non-empty HTML report.
