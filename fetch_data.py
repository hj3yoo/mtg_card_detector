from urllib import request, error
import ast
import json
import pandas as pd
import re
import os
import transform_data
import time


# Core & expansion sets with 2003 frame
set_2003_list = ['mrd', 'dst', '5dn', 'chk', 'bok', 'sok', 'rav', 'gpt', 'dis', 'csp', 'tsp', 'plc', 'fut', '10e',
                 'lrw', 'mor', 'shm', 'eve', 'ala', 'con', 'arb', 'm10', 'zen', 'wwk', 'roe', 'm11', 'som', 'mbs',
                 'nph', 'm12', 'isd', 'dka', 'avr', 'm13', 'rtr', 'gtc', 'dgm', 'm14', 'ths', 'bng', 'jou']
# Core & expansion sets with 2015 frame
set_2015_list = ['m15', 'ktk', 'frf', 'dtk', 'bfz', 'ogw', 'soi', 'emn', 'kld', 'aer', 'akh', 'hou', 'xln', 'rix', 'dom']

# Box sets
set_box_list = ['evg', 'drb', 'dd2', 'ddc', 'td0', 'v09', 'ddd', 'h09', 'dde', 'dpa', 'v10', 'ddf', 'td0', 'pd2', 'ddg',
                'cmd', 'v11', 'ddh', 'pd3', 'ddi', 'v12', 'ddj', 'cm1', 'td2', 'ddk', 'v13', 'ddl', 'c13', 'ddm', 'md1',
                'v14', 'ddn', 'c14', 'ddo', 'v15', 'ddp', 'c15', 'ddq', 'v16', 'ddr', 'c16', 'pca', 'dds', 'cma', 'c17',
                'ddt', 'v17', 'ddu', 'cm2', 'ss1', 'gs1', 'c18']

# Supplemental sets
set_sup_list = ['hop', 'arc', 'pc2', 'cns', 'cn2', 'e01', 'e02', 'bbd']

all_set_list = set_2003_list


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
        #df = df[['artist', 'border_color', 'collector_number', 'color_identity', 'colors', 'flavor_text', 'image_uris',
        #         'mana_cost', 'legalities', 'name', 'oracle_text', 'rarity', 'type_line', 'set', 'set_name', 'power',
        #         'toughness']]
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
    if out_dir == '':
        out_dir = '%s/card_img/%s/%s' % (transform_data.data_dir, size, row['set'])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    png_urls = []
    card_names = []
    if row['layout'] == 'transform' or row['layout'] == 'double_faced_token':
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
    for set_name in all_set_list:
        csv_name = '%s/csv/%s.csv' % (transform_data.data_dir, set_name)
        print(csv_name)
        if not os.path.isfile(csv_name):
            df = fetch_all_cards_text(url='https://api.scryfall.com/cards/search?q=set:%s+lang:en'
                                          % set_name, csv_name=csv_name)
        else:
            df = load_all_cards_text(csv_name)
        df.sort_values('collector_number')
        fetch_all_cards_image(df, out_dir='%s/card_img/png/%s' % (transform_data.data_dir, set_name))
    #df = fetch_all_cards_text(url='https://api.scryfall.com/cards/search?q=layout:normal+lang:en+frame:2003',
    #                          csv_name='data/csv/all.csv')
    pass


if __name__ == '__main__':
    main()
    pass
