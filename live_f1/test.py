import asyncio
import json
import os
import requests
import websockets
from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv()

COOKIE_VALUE = os.getenv("F1_SESSION_COOKIE")
if not COOKIE_VALUE:
    raise ValueError("F1_SESSION_COOKIE missing. Check your .env file.")

BASE_URL = "https://livetiming.formula1.com/signalr"
CONNECTION_DATA = "%5B%7B%22name%22%3A%22Streaming%22%7D%5D"

HDR = {"User-Agent": "BestHTTP", "Accept-Encoding": "gzip,identity"}

TOPICS = [
    "Heartbeat", "CarData.z", "Position.z", "ExtrapolatedClock",
    "TopThree", "TimingStats", "TimingAppData", "WeatherData",
    "TrackStatus", "DriverList", "RaceControlMessages",
    "SessionInfo", "SessionData", "LapCount", "TimingData",
]


def negotiate():
    headers = {**HDR, "Cookie": f"login-session={COOKIE_VALUE}"}
    url = f"{BASE_URL}/negotiate?clientProtocol=1.5&connectionData={CONNECTION_DATA}"
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    return r.json()["ConnectionToken"], r.cookies.get("AWSALBCORS")


async def connect_and_stream(token, aws_cookie):
    ws_url = (
        f"wss://livetiming.formula1.com/signalr/connect"
        f"?clientProtocol=1.5&transport=webSockets"
        f"&connectionToken={quote(token)}"
        f"&connectionData={CONNECTION_DATA}"
    )

    cookie_str = f"login-session={COOKIE_VALUE}"
    if aws_cookie:
        cookie_str += f"; AWSALBCORS={aws_cookie}"
    headers = {**HDR, "Cookie": cookie_str}

    async with websockets.connect(ws_url, additional_headers=headers) as ws:
        print("WebSocket connected.")

        sub_msg = {
            "H": "Streaming",
            "M": "Subscribe",
            "A": [TOPICS],
            "I": 1,
        }
        await ws.send(json.dumps(sub_msg))
        print(f"Subscribed to {len(TOPICS)} topics.\n")

        while True:
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=15.0)
            except asyncio.TimeoutError:
                print("idle...")
                continue
            except websockets.ConnectionClosed:
                print("Connection closed.")
                break

            try:
                m = json.loads(message)
            except json.JSONDecodeError:
                print(f"[non-json] {message[:120]}")
                continue

            if m == {}:
                continue  # ping

            if "R" in m:
                topics = list(m["R"].keys())
                print(f">>> SNAPSHOT TOPICS ({len(topics)}): {topics}")
                auth_topics = {"CarData.z", "Position.z"} & set(topics)
                if auth_topics:
                    print(f">>> AUTH OK: {auth_topics}")
                else:
                    print(">>> auth-only topics absent (either no session live, or auth rejected)")
                print()

            elif "M" in m and m["M"]:
                for item in m["M"]:
                    if "A" in item and item["A"]:
                        print(f"[FEED] {item['A'][0]}")


if __name__ == "__main__":
    print("Negotiating...")
    token, aws_cookie = negotiate()
    print(f"Token: {token[:30]}...  AWSALBCORS: {'yes' if aws_cookie else 'no'}\n")
    asyncio.run(connect_and_stream(token, aws_cookie))