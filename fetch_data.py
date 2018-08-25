from urllib import request
import json
import pandas as pd
import re


def fetch_all_cards_text(url='https://api.scryfall.com/cards/search?q=layout:normal+format:modern+lang:en+frame:2003',
                         csv_name=''):
    has_more = True
    cards = []
    while has_more:
        res_file_dir, http_message = request.urlretrieve(url)
        with open(res_file_dir) as res_file:
            res_json = json.loads(res_file.read())
            cards += res_json['data']
            has_more = res_json['has_more']
            if has_more:
                url = res_json['next_page']
            print(len(cards))

    df = pd.DataFrame.from_dict(cards)
    df['image'] = ''
    for ind, row in df.iterrows():
        df.set_value(ind, 'image', row['image_uris']['png'])

    if csv_name != '':
        df = df[['artist', 'border_color', 'collector_number', 'color_identity', 'colors', 'flavor_text', 'image_uris',
                 'image', 'mana_cost', 'legalities', 'name', 'oracle_text', 'rarity', 'type_line', 'set', 'set_name',
                 'power', 'toughness']]
        df.to_csv(csv_name, sep=';')

    return cards


def get_valid_filename(s):
    """
    NOTE: Pulled from Django framework (https://github.com/django/django/blob/master/django/utils/text.py)
    Return the given string converted to a string that can be used for a clean
    filename. Remove leading and trailing spaces; convert other spaces to
    underscores; and remove anything that is not an alphanumeric, dash,
    underscore, or dot.
    >>> get_valid_filename("john's portrait in 2004.jpg")
    'johns_portrait_in_2004.jpg'
    """
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)


def fetch_cards_image(cards_json, out_dir, size='large'):
    for card in cards_json:
        request.urlretrieve(card['image_uris'][size], '%s\%s' % (out_dir, card['name']))
    pass


def main():
    fetch_all_cards_text(csv_name='data/all_cards.csv')

if __name__ == '__main__':
    main()
