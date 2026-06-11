import requests, sys
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding="utf-8")

r = requests.get("https://www.yeastract.com/consensuslist.php", timeout=30)
soup = BeautifulSoup(r.content, "html.parser")

# Table 5 (class=center) is the data table
data_table = soup.find("table", {"class": "center"})
rows = data_table.find_all("tr")
print(f"Total rows: {len(rows)}")

records = []
for row in rows:
    cells = row.find_all("td")
    if len(cells) >= 2:
        tf = cells[0].get_text(strip=True)
        consensus = cells[1].get_text(strip=True)
        if tf and consensus and not tf.startswith("Transcription"):
            records.append((tf, consensus))

print(f"Records parsed: {len(records)}")
unique_tfs = sorted(set(r[0] for r in records))
print(f"Unique TFs: {len(unique_tfs)}")
print("TF names:", unique_tfs[:30])
print()
print("First 10 records:")
for tf, seq in records[:10]:
    print(f"  {tf:20s} {seq}")
