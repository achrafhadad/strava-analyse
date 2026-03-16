"""
Strava OAuth2 authenticatie.
Opent een browser voor autorisatie en slaat tokens lokaal op.
"""
import json
import os
import time
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import requests

TOKEN_FILE = os.path.join(os.path.dirname(__file__), "tokens.json")
ENV_FILE = os.path.join(os.path.dirname(__file__), ".env")

def load_env():
    """Laad client_id en client_secret uit .env bestand."""
    if not os.path.exists(ENV_FILE):
        raise FileNotFoundError(
            "Maak een .env bestand aan met STRAVA_CLIENT_ID en STRAVA_CLIENT_SECRET.\n"
            "Zie .env.example voor het format."
        )
    env = {}
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip()
    return env["STRAVA_CLIENT_ID"], env["STRAVA_CLIENT_SECRET"]


class CallbackHandler(BaseHTTPRequestHandler):
    """Vangt de OAuth callback op."""
    authorization_code = None

    def do_GET(self):
        query = parse_qs(urlparse(self.path).query)
        if "code" in query:
            CallbackHandler.authorization_code = query["code"][0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<h1>Autorisatie gelukt! Je kunt dit venster sluiten.</h1>")
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"<h1>Autorisatie mislukt.</h1>")

    def log_message(self, format, *args):
        pass  # Stille server


def authenticate():
    """Volledige OAuth2 flow: autorisatie + token exchange."""
    client_id, client_secret = load_env()

    auth_url = (
        f"https://www.strava.com/oauth/authorize"
        f"?client_id={client_id}"
        f"&response_type=code"
        f"&redirect_uri=http://localhost:8089/callback"
        f"&approval_prompt=auto"
        f"&scope=read_all,activity:read_all"
    )

    print("Browser wordt geopend voor Strava autorisatie...")
    webbrowser.open(auth_url)

    server = HTTPServer(("localhost", 8089), CallbackHandler)
    server.handle_request()

    if not CallbackHandler.authorization_code:
        raise Exception("Geen autorisatiecode ontvangen.")

    # Token exchange
    response = requests.post("https://www.strava.com/oauth/token", data={
        "client_id": client_id,
        "client_secret": client_secret,
        "code": CallbackHandler.authorization_code,
        "grant_type": "authorization_code",
    })
    response.raise_for_status()
    tokens = response.json()

    save_tokens(tokens)
    print("Authenticatie geslaagd!")
    return tokens["access_token"]


def save_tokens(tokens):
    """Sla tokens op in lokaal bestand."""
    data = {
        "access_token": tokens["access_token"],
        "refresh_token": tokens["refresh_token"],
        "expires_at": tokens["expires_at"],
    }
    with open(TOKEN_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_access_token():
    """Haal een geldig access token op, vernieuw indien nodig."""
    if not os.path.exists(TOKEN_FILE):
        return authenticate()

    with open(TOKEN_FILE) as f:
        tokens = json.load(f)

    # Token verlopen? Vernieuw het.
    if time.time() >= tokens["expires_at"]:
        client_id, client_secret = load_env()
        response = requests.post("https://www.strava.com/oauth/token", data={
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": tokens["refresh_token"],
            "grant_type": "refresh_token",
        })
        response.raise_for_status()
        new_tokens = response.json()
        save_tokens(new_tokens)
        return new_tokens["access_token"]

    return tokens["access_token"]


if __name__ == "__main__":
    token = authenticate()
    print(f"Access token verkregen (eerste 10 chars): {token[:10]}...")
