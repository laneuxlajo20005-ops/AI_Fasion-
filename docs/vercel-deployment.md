# Deploying to Vercel

The Flask application is exposed through `api/index.py`. Local development
and isolated unit tests can use SQLite. Production requires `DATABASE_URL` and
uses PostgreSQL for both named profiles and temporary wizard state; startup
refuses to fall back to a local file when production is detected. Vercel sets
`VERCEL=1`; other production hosts should set `APP_ENV=production`.
The repository pins Vercel's supported Python 3.12 runtime in `.python-version`.

## Database initialization and migrations

The versioned migration runner creates `schema_migrations`, then applies any
pending schema changes transactionally. It is safe to run at startup and uses
a PostgreSQL transaction-scoped advisory lock to serialize simultaneous cold
starts. Existing compatible profile and wizard tables are retained when the
first migration is recorded. Migrations create schema only: they do not copy
profile data from a developer's ignored local SQLite database.

Set `DATABASE_URL` in PowerShell without putting its value in source control:

```powershell
$env:DATABASE_URL = "postgresql://USER:PASSWORD@HOST/DBNAME?sslmode=require"
python -m database migrate
```

Configure `DATABASE_URL` in the shell environment from a secure source before
running these commands; do not type a real connection string into a command
that will be saved in terminal history. Use Neon’s pooled connection URL for
serverless traffic and its TLS settings (`sslmode=require`).
The application also runs pending migrations during startup. PostgreSQL
connection errors fail startup with a sanitized message; database failures
during requests return HTTP 503.

To run the PostgreSQL integration test against the configured database without
putting the URL in test output:

```powershell
$env:TEST_DATABASE_URL = $env:DATABASE_URL
python -m unittest tests.test_database.DatabaseTests.test_postgresql_profile_and_wizard_state_persist_across_store_instances -v
Remove-Item Env:TEST_DATABASE_URL
```

The test creates a uniquely identified profile and wizard record, reads them
through newly constructed store instances, then removes those test rows.

## GitHub and Vercel

1. Push the project to a GitHub repository, then import that repository in
   Vercel. Keep the project root set to the directory containing `vercel.json`
   and `requirements.txt`.
2. Create a Neon PostgreSQL database and copy its connection URL into a secret
   environment variable in Vercel. Use Neon TLS settings (`sslmode=require` or
   a stronger verification mode).
3. Add these Vercel environment variables for the Production environment:
   - `DATABASE_URL`: the provider's PostgreSQL connection URL, including its
     TLS/SSL options.
   - `FLASK_SECRET_KEY`: a long random secret, for example the output of
     `python -c "import secrets; print(secrets.token_hex(32))"`.
   - `FLASK_COOKIE_SECURE`: `1`.
4. Run `python -m database migrate` locally against the Neon URL once, or rely
   on the transactional startup migration. Add the same variables to Preview
   if preview deployments
   should use a database and persistent session signing key.

Do not commit `.env` files, database credentials, or the local SQLite database.
PostgreSQL connection failures are reported during app startup rather than
silently falling back to ephemeral local storage.

The Vercel `requirements.txt` intentionally excludes MediaPipe/OpenCV and
offline-only Matplotlib/joblib to stay within the Python function bundle limit.
Consequently, image-based body analysis is unavailable on Vercel; the rest of
the recommendation flow remains independent of image upload. To run the full
computer-vision project locally, install `requirements-full.txt`. Uploaded
photos remain transient and are not saved. Weather requests require outbound
network access.

Vercel currently supports Python 3.12, 3.13, and 3.14; this project pins 3.12.
The Linux wheel check against that runtime should be repeated if native
dependency versions change. A resolved full dependency set
including MediaPipe/OpenCV expands to roughly 939 MB of uncompressed wheel
contents in the current project, exceeding Vercel's documented 500 MB
uncompressed function bundle limit. The deployment-only dependency set
resolved successfully to Linux Python 3.12 wheels and has roughly 253 MB of
uncompressed wheel contents. This estimate is below the limit but is not a
measured Vercel build size; an actual deployment is still required to validate
the final platform bundle and runtime behavior.

References: [Vercel Python versions](https://vercel.com/docs/functions/runtimes/python/python-version)
and [function bundle limits](https://vercel.com/docs/functions/limitations#bundle-size-limits).
