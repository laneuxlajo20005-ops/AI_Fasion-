# Deploying to Vercel

The Flask application is exposed through `api/index.py`. Local development
continues to use SQLite; setting `DATABASE_URL` switches saved profiles and
temporary wizard state to PostgreSQL. The tables are created automatically
when the application starts. Local SQLite data is not copied to PostgreSQL.

## GitHub and Vercel

1. Push the project to a GitHub repository, then import that repository in
   Vercel. Keep the project root set to the directory containing `vercel.json`
   and `requirements.txt`.
2. Create a PostgreSQL database with a provider such as Neon or Supabase.
3. Add these Vercel environment variables for the Production environment:
   - `DATABASE_URL`: the provider's PostgreSQL connection URL, including its
     TLS/SSL options.
   - `FLASK_SECRET_KEY`: a long random secret, for example the output of
     `python -c "import secrets; print(secrets.token_hex(32))"`.
   - `FLASK_COOKIE_SECURE`: `1`.
4. Deploy from Vercel. Add the same variables to Preview if preview deployments
   should use a database and persistent session signing key.

Do not commit `.env` files, database credentials, or the local SQLite database.
PostgreSQL connection failures are reported during app startup rather than
silently falling back to ephemeral local storage.

Uploaded photos remain transient and are not saved. Weather requests require
outbound network access. Vercel build and function-size limits may constrain
the computer-vision and scientific Python dependencies; check the deployment
build logs if the platform rejects the generated function bundle.
