import requests
import json

url = "http://127.0.0.1:11434/api/generate"
payload = {
    "model": "phi3:mini",
    "prompt": "Summarise Stake Casino in 50 words.",
    "stream": True,
    "options": {
        "temperature": 0.1,
        "num_predict": 80
    }
}

with requests.post(url, json=payload, stream=True, timeout=60) as r:
    r.raise_for_status()
    for line in r.iter_lines(decode_unicode=False):
        if not line:
            continue
        line = line.decode("utf-8", errors="ignore")

        if line.startswith("data:"):
            obj = json.loads(line[5:].strip())
            print(obj)

            if obj.get("done"):
                print("Done!")
                break
