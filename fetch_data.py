import ast
import json
import os
import pandas as pd
import re
from urllib import request, error

from config import Config

"""
Note: All codes in this file realies on Scryfall API to aggregate card database and their images.
Scryfall API doc is available at: https://scryfall.com/docs/api
"""


def fetch_all_cards_text(url='https://api.scryfall.com/cards/search?q=layout:normal+format:modern+lang:en+frame:2003',
                         csv_name=None):
    """
    Given the query URL using Scryfall API, aggregate all card information and convert them from json to table
    :param url: query URL
    :param csv_name: path of the csv file to save the result
    :return: pandas dataframe of the fetch cards
    """
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

    if csv_name is not None:
        #df = df[['artist', 'border_color', 'collector_number', 'color_identity', 'colors', 'flavor_text', 'image_uris',
        #         'mana_cost', 'legalities', 'name', 'oracle_text', 'rarity', 'type_line', 'set', 'set_name', 'power',
        #         'toughness']]
        df.to_csv(csv_name, sep=';')  # Comma seperator doesn't work, since some columns are saved as a dict
    return df


def load_all_cards_text(csv_name):
    df = pd.read_csv(csv_name, sep=';')   # Comma seperator doesn't work, since some columns are saved as a dict
    return df


def get_valid_filename(s):
    """
    Return the given string converted to a string that can be used for a clean
    filename. Remove leading and trailing spaces; convert other spaces to
    underscores; and remove anything that is not an alphanumeric, dash,
    underscore, or dot.
    >>> get_valid_filename("john's portrait in 2004.jpg")
    'johns_portrait_in_2004.jpg'
    From: https://github.com/django/django/blob/master/django/utils/text.py
    :param s: input string
    :return: string of valid filename
    """
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)


def fetch_all_cards_image(df, out_dir=None, size='png'):
    """
    Download card images from Scryfall database
    :param df: pandas dataframe (or series) of cards
    :param out_dir: path of output directory
    :param size: Image format given by Scryfall API - 'png', 'large', 'normal', 'small', 'art_crop', 'border_crop'
    :return:
    """
    if size != 'png':
        print('Note: this repo has been implemented using only \'png\' size. '
              'Using %s may result in an unexpected behaviour in other parts of this repo.' % size)
    if isinstance(df, pd.Series):
        # df is a single row of card
        fetch_card_image(df, out_dir, size)
    else:
        # df is a dataframe containing list of cards
        for ind, row in df.iterrows():
            fetch_card_image(row, out_dir, size)


def fetch_card_image(row, out_dir=None, size='png'):
    """
    Download a card's image from Scryfall database
    :param row: pandas series including the card's information
    :param out_dir: path of the output directory
    :param size: Image format given by Scryfall API - 'png', 'large', 'normal', 'small', 'art_crop', 'border_crop'
    :return:
    """
    if out_dir is None:
        out_dir = '%s/card_img/%s/%s' % (Config.data_dir, size, row['set'])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Extract card's name and URL for image accordingly
    # Double-faced cards have a different format, and results in two separate card images
    png_urls = []
    card_names = []
    if row['layout'] in ['transform', 'double_faced_token']:
        if isinstance(row['card_faces'], str):  # For some reason, dict isn't being parsed in the previous step
            card_faces = ast.literal_eval(row['card_faces'])
        else:
            card_faces = row['card_faces']
        for i in range(len(card_faces)):
            png_urls.append(card_faces[i]['image_uris'][size])
            card_names.append(get_valid_filename(card_faces[i]['name']))
    else: #if row['layout'] == 'normal':
        if isinstance(row['image_uris'], str):  # For some reason, dict isn't being parsed in the previous step
            png_urls.append(ast.literal_eval(row['image_uris'])[size])
        else:
            png_urls.append(row['image_uris'][size])
        card_names.append(get_valid_filename(row['name']))

    for i in range(len(png_urls)):
        img_name = '%s/%s_%s.png' % (out_dir, row['collector_number'], card_names[i])
        if not os.path.isfile(img_name):
            request.urlretrieve(png_urls[i], filename=img_name)
            print(img_name)


def main():
    # Query card data by each set, then merge them together
    for set_name in Config.all_set_list:
        csv_name = '%s/csv/%s.csv' % (Config.data_dir, set_name)
        print(csv_name)
        if not os.path.isfile(csv_name):
            df = fetch_all_cards_text(url='https://api.scryfall.com/cards/search?q=set:%s+lang:en' % set_name,
                                      csv_name=csv_name)
        else:
            df = load_all_cards_text(csv_name)
        df.sort_values('collector_number')
        fetch_all_cards_image(df, out_dir='%s/card_img/png/%s' % (Config.data_dir, set_name))

    #df = fetch_all_cards_text(url='https://api.scryfall.com/cards/search?q=layout:normal+lang:en+frame:2003',
    #                          csv_name='%s/csv/all.csv' % Config.data_dir)
    return


if __name__ == '__main__':
    main()
