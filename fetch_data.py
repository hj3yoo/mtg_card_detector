from urllib import request
import ast
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
        with open(res_file_dir, 'r') as res_file:
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
        #df.to_json(csv_name)
        df.to_csv(csv_name, sep=';')  # Comma doesn't work, since some columns are saved as a dict

    return df


def load_all_cards_text(csv_name):
    #with open(csv_name, 'r') as json_file:
    #    cards = json.loads(json_file.read())
    #df = pd.DataFrame.from_dict(cards)
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


def fetch_all_cards_image(df, out_dir='', size='png'):
    if isinstance(df, pd.Series):
        fetch_card_image(df, out_dir, size)
    else:
        for ind, row in df.iterrows():
            fetch_card_image(row, out_dir, size)


def fetch_card_image(row, out_dir='', size='png'):
    if isinstance(row['image_uris'], str):  # For some reason, dict isn't being parsed in the previous step
        png_url = ast.literal_eval(row['image_uris'])[size]
    else:
        png_url = row['image_uris'][size]
    if out_dir == '':
        out_dir = 'data/%s/%s' % (size, row['set'])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    img_name = '%s/%s_%s.png' % (out_dir, row['collector_number'], get_valid_filename(row['name']))
    if not os.path.isfile(img_name):
        request.urlretrieve(png_url, filename=img_name)
        print(img_name)


def main():
    for set_name in ['8ed', 'mrd', 'dst', '5dn', 'chk', 'bok', 'sok', '9ed', 'rav', 'gpt', 'dis', 'csp', 'tsp', 'plc',
                     'fut', '10e', 'lrw', 'mor', 'shm', 'eve', 'ala', 'con', 'arb', 'm10', 'zen', 'wwk', 'roe', 'm11',
                     'som', 'mbs', 'nph', 'm12', 'isd', 'dka', 'avr', 'm13', 'rtr', 'gtc', 'dgm', 'm14', 'ths', 'bng',
                     'jou']:
        csv_name = 'data/csv/%s.csv' % set_name
        if not os.path.isfile(csv_name):
            df = fetch_all_cards_text(url='https://api.scryfall.com/cards/search?q=layout:normal+set:%s+lang:en' % set_name,
                                      csv_name=csv_name)
        else:
            df = load_all_cards_text(csv_name)
        print(csv_name)
        fetch_all_cards_image(df, out_dir='../usb/data/png/%s' % set_name)
    pass


if __name__ == '__main__':
    main()
    pass
