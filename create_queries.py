categories = [
    "tübingen attractions", "food and drinks", "restaurants", "cafes", "bars", "hotels", "university", "student life",
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
        query_text = f"{topic}"
        f.write(f"{i+1}\t{query_text}\n")
