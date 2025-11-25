import asyncio
import json
from tcgdexsdk import TCGdex, Query
from tcgdexsdk.enums import Quality
import requests
from tqdm import tqdm
import os

def pricing(id):
    url = f"https://api.tcgdex.net/v2/en/cards/{id}"
    response = requests.get(url)

    if not response.ok:
        return -1

    data = response.json()
    if data["pricing"] is not None:
        if data["pricing"]["cardmarket"] is not None:
            if data["pricing"]["cardmarket"]["avg"] is not None:
                return float(data["pricing"]["cardmarket"]["avg"])
    return -1

async def main():
    tcgdex = TCGdex("de")
    p = 0
    while True:
        page = await tcgdex.card.list(Query().paginate(page=p, itemsPerPage=5000))
        print("Got", len(page), "cards on page", p)
        if (len(page) == 0):
            print("No nore cards in page. Exiting ...")
            break

        for card in tqdm(page):
            if os.path.exists("data/"+card.id+".json"):
                continue
            try:
                cd = await tcgdex.card.get(card.id)
                #print(cd.name, cd.image)
                image_url = cd.get_image_url(quality=Quality.HIGH, extension="png")
                if image_url:
                    image_file = "data/" + card.id + '.jpg'
                    with open(image_file, 'wb') as handle:
                        response = requests.get(image_url, stream=True)

                        if not response.ok:
                            print(response)
                        else:
                            for block in response.iter_content(1024):
                                if not block:
                                    break

                                handle.write(block)

                            price = pricing(cd.id)
                            print("Price: ", price)

                            j = {
                                "id": cd.id,
                                "name": cd.name,
                                "image": image_file,
                                "rarity": cd.rarity,
                                "hp": cd.hp,
                                "price": price
                            }
                            with open("data/" + card.id + ".json", "w", encoding="utf-8") as f:
                                json.dump(j, f)
                else:
                    #print("No image in ", cd)
                    pass
            except Exception as e:
                print("Error while reading card", card.id, e)

        p += 1
asyncio.run(main())