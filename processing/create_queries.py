import requests
import time
from keys import API_KEY, CSE_ID

# creates queries.txt file with 100 query entries with the categories below
# additionally creates qrels.txt file containing top 10 websites to each topic according to Google API
# can be increased to 100 top results via num_results parameter


categories = [
    "attractions", "food and drinks", "restaurants", "cafes", "bars", "hotels", "university", "student life",
    "museums", "architecture", "history", "weather", "transportation", "bike rentals",
    "day trips", "nature", "hiking trails", "events", "theater", "art galleries", "shopping",
    "souvenirs", "local cuisine", "vegan food", "bakeries", "public transport",
    "train station", "nightlife", "parks", "castles", "churches", "riverside walk",
    "Neckar river", "market square", "Christmas market", "summer festivals", "wine tasting",
    "traditional food", "student housing", "dorms", "libraries", "concerts", "bus schedule",
    "bike paths", "Tübingen University", "study abroad", "international students", "visa help",
    "expat life", "job opportunities", "internships", "city map", "photography spots",
    "walking tours", "city guide", "language schools", "living cost", "budget travel",
    "local news", "weather forecast", "best views", "hidden gems", "parking", "airports near",
    "travel tips", "shopping streets", "second-hand stores", "organic food", "brew pubs",
    "German courses", "cultural life", "open-air cinema", "student card", "biking routes",
    "rental apartments", "cost of living", "cooking classes", "walking trails", "student clubs",
    "local festivals", "visa registration", "Tübingen nightlife", "pubs", "study tips",
    "research institutes", "cinemas", "roommates", "climate", "expat tips",
    "youth hostels", "city bus", "timetable", "travel card", "local radio", "post office",
    "grocery stores", "biking safety", "guided tours", "weekend ideas", "surrounding towns"
]

with open("queries.txt", "w", encoding="utf-8") as f:
    for i, topic in enumerate(categories[:100]):
        query_text = f"tübingen {topic}"
        f.write(f"{i+1}\t{query_text}\n")


API_KEY = API_KEY
CSE_ID = CSE_ID

def google_search(query, num_results=10, sleep_between=1.0):
    results = []
    for start in range(1, num_results + 1, 10):
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "q": query,
            "key": API_KEY,
            "cx": CSE_ID,
            "start": start,
            "num": min(10, num_results - len(results)),
        }
        r = requests.get(url, params=params)
        data = r.json()
        for item in data.get("items", []):
            results.append(item["link"])
        time.sleep(sleep_between)  # avoid hitting rate limit
    return results

with open("queries.txt", "r", encoding="utf-8") as f, open("qrels.txt", "w", encoding="utf-8") as out:
    for line in f:
        qid, query = line.strip().split("\t")
        urls = google_search(query, num_results=10)
        for url in urls:
            out.write(f"{qid}\t0\t{url}\t1\n")

