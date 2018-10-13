import os


class Config:
    # List of all black-bordered cards printed from 8th edition and onwards (8ed and 9ed are white-bordered)
    # Core & expansion sets with 2003 frame
    set_2003_list = ['mrd', 'dst', '5dn', 'chk', 'bok', 'sok', 'rav', 'gpt', 'dis', 'csp', 'tsp', 'plc', 'fut', '10e',
                     'lrw', 'mor', 'shm', 'eve', 'ala', 'con', 'arb', 'm10', 'zen', 'wwk', 'roe', 'm11', 'som', 'mbs',
                     'nph', 'm12', 'isd', 'dka', 'avr', 'm13', 'rtr', 'gtc', 'dgm', 'm14', 'ths', 'bng', 'jou']
    # Core & expansion sets with 2015 frame
    set_2015_list = ['m15', 'ktk', 'frf', 'dtk', 'bfz', 'ogw', 'soi', 'emn', 'kld', 'aer', 'akh', 'hou', 'xln', 'rix',
                     'dom']
    # Box sets
    set_box_list = ['evg', 'drb', 'dd2', 'ddc', 'td0', 'v09', 'ddd', 'h09', 'dde', 'dpa', 'v10', 'ddf', 'td0', 'pd2',
                    'ddg',
                    'cmd', 'v11', 'ddh', 'pd3', 'ddi', 'v12', 'ddj', 'cm1', 'td2', 'ddk', 'v13', 'ddl', 'c13', 'ddm',
                    'md1',
                    'v14', 'ddn', 'c14', 'ddo', 'v15', 'ddp', 'c15', 'ddq', 'v16', 'ddr', 'c16', 'pca', 'dds', 'cma',
                    'c17',
                    'ddt', 'v17', 'ddu', 'cm2', 'ss1', 'gs1', 'c18']
    # Supplemental sets
    set_sup_list = ['hop', 'arc', 'pc2', 'cns', 'cn2', 'e01', 'e02', 'bbd']
    all_set_list = set_2003_list  #+ set_2015_list + set_box_list + set_sup_list

    card_mask_path = os.path.abspath('data/mask.png')
    data_dir = os.path.abspath('/media/win10/data')
    darknet_dir = os.path.abspath('.')