from urllib import request
import json
import pandas as pd
import re
import os


def fetch_all_cards_text(url='https://api.scryfall.com/cards/search?q=layout:normal+format:modern+lang:en+frame:2003',
                         csv_name=''):
    has_more = True
    cards = []
    # get cards dataset as a json from the query
    while has_more:
        res_file_dir, http_message = request.urlretrieve(url)
        with open(res_file_dir) as res_file:
            res_json = json.loads(res_file.read())
            cards += res_json['data']
            has_more = res_json['has_more']
            if has_more:
                url = res_json['next_page']
            print(len(cards))

    # Convert them into a dataframe, and truncate unnecessary columns
    df = pd.DataFrame.from_dict(cards)

    if csv_name != '':
        df = df[['artist', 'border_color', 'collector_number', 'color_identity', 'colors', 'flavor_text', 'image_uris',
                 'mana_cost', 'legalities', 'name', 'oracle_text', 'rarity', 'type_line', 'set', 'set_name', 'power',
                 'toughness']]
        df.to_csv(csv_name, sep=';')  # Comma doesn't work, since some columns are saved as a dict

    return df


def load_all_cards_text(csv_name):
    df = pd.read_csv(csv_name, sep=';')
    return df


# Pulled from Django framework (https://github.com/django/django/blob/master/django/utils/text.py)
def get_valid_filename(s):
    """
    Return the given string converted to a string that can be used for a clean
    filename. Remove leading and trailing spaces; convert other spaces to
    underscores; and remove anything that is not an alphanumeric, dash,
    underscore, or dot.
    >>> get_valid_filename("john's portrait in 2004.jpg")
    'johns_portrait_in_2004.jpg'
    """
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)


def fetch_cards_image(df, out_dir='', size='png'):
    for ind, row in df.iterrows():
        png_url = row['image_uris'][size]
        if out_dir == '':
            out_dir = 'data/%s/%s' % (size, row['set'])
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        img_name = '%s/%s_%s.png' % (out_dir, row['collector_number'], get_valid_filename(row['name']))
        request.urlretrieve(png_url, filename=img_name)
        print(img_name)
    pass


def main():
    df = fetch_all_cards_text(url='https://api.scryfall.com/cards/search?q=layout:normal+set:rtr+lang:en',
                              csv_name='data/all_cards.csv')
    #fetch_cards_image(df)
    pass


if __name__ == '__main__':
    main()
    pass
