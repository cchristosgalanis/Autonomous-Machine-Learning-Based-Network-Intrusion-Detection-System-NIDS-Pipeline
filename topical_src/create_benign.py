import requests
import time
import random

urls = [
    "https://el.wikipedia.org",
    "https://www.github.com",
    "https://www.google.com",
    "https://www.reddit.com"
]

print("Starting benign web traffic generation...")
while True:
    target = random.choice(urls)
    try:
        requests.get(target, timeout=5)
        print(f"Visited {target}")
    except:
        pass
    # Περιμένει τυχαία από 1 έως 5 δευτερόλεπτα (μοιάζει με ανθρώπινη συμπεριφορά)
    time.sleep(random.randint(1, 5))