from urllib import request


def fetch_all_cards_text(url='https://archive.scryfall.com/json/scryfall-all-cards.json', out_file='all_cards.json'):
    request.urlretrieve(url, out_file)
    pass

def fetch_cards_image(cards_json, out_dir, size='large'):
    for card in cards_json:
        request.urlretrieve(card['image_uris'][size], '%s\%s' % (out_dir, card['name']))
    pass