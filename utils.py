"""
General tasks for inMySky
"""

import os
import argparse as ap
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from datetime import (
    datetime, 
    timedelta
)

from st.tle import TLE
from st.site import Site

# necessary paths
OUT_DIR = './data/'
SATCAT_PATH = '{}sat_cat.csv'.format(OUT_DIR)
TLECAT_PATH = '{}tle_cat.txt'.format(OUT_DIR)  # update if paths don't follow standard format

# labels for onPick function
LABELS = [('Name', 'NAME'),
          ('ID', 'NORAD_ID'),
          ('Type', 'TYPE'),
          ('Country', 'COUNTRY'),
          ('Size', 'SIZE'),
          ('Launch', 'LAUNCH_YR'),
          ('Incl', 'INCLINATION')]

class Log:
    """
    Convenience class for log-keeping
    """
    def __init__(self, n_rows, col_names, col_dtype):
        """
        Initialise Log
        
        Parameters
        ----------
        n_rows : int
            Number of rows for log
        col_names : list
            List of column names, e.g. ['col0', 'col1']
        col_dtype : list
            List of column dtypes, e.g. ['U25', 'f8']
        """
        self.table = Table(np.zeros((n_rows, len(col_names))),
                           names=col_names,
                           dtype=col_dtype)

    def update(self, column, row, entry):
        """
        Update cell within log
        
        Parameters
        ----------
        column : str
            Name of column containing cell
        row : int
            Index of row containing cell
        entry : int, float, str
            Desired cell entry
        """
        self.table[column][row] = entry

    def add_row(self, entry):
        """
        Add row to log
        
        Parameters
        ----------
        entry : List
            List of cell entries to append to log
        """
        self.table.add_row(entry)

def argParse():
    """
    Argument parser settings
    
    Parameters
    ----------
    None
    
    Returns
    -------
    args : array-like
        Array of command line arguments
    """
    parser = ap.ArgumentParser()
    
    parser.add_argument('--data_dir',
                        help='path to directory containing satcat and TLE catalogues',
                        type=str,
                        default='./data/')
    
    parser.add_argument('--epoch',
                        help='epoch for TLE propagation, format: YYYY-mm-ddTHH:MM:SS.s [utc]',
                        type=str,
                        default='now')
    
    parser.add_argument('--site',
                        help='observation site for TLE propagation, e.g. "INT"\n'
                             'NB: new sites can be added to st.site',
                        type=str,
                        default='INT')
    
    return parser.parse_args()

def parseInput():
    """
    Parse user input arguments
    
    Parameters
    ----------
    None
    
    Returns
    -------
    epoch : datetime.datetime
        Epoch for TLE propagation
    site : st.site.Site
        Observation site for TLE propagation
    """
    args = argParse()
        
    if args.epoch != 'now':
        try:
            epoch = datetime.strptime(args.epoch, '%Y-%m-%dT%H:%M:%S.%f')
        except:
            print('Required format for epoch: "YYYY-mm-ddTHH:MM:SS.ff"')
            quit()
        else:
            print('Warning: TLE accuracy deteriorates rapidly!')
    else:
        epoch = datetime.now()
     
    return epoch, Site(args.site)

def loadCats(sat_path, tle_path):
    """
    Load satcat and TLE catalogue
    
    Parameters
    ----------
    sat_path, tle_path : str
        Paths to satcat and TLE catalogues
    
    Returns
    -------
    sat_cat : astropy.table.Table
        Table containing satcat information
    tle_cat : array-like
        List of TLEs [st.tle.TLE]
    """
    # load satcat catalogue
    if os.path.exists(SATCAT_PATH):
        sat_cat = Table.read(SATCAT_PATH, format='csv')
    else:
        print('Invalid path to satcat catalogue...')
        quit()
    
    # load tle catalogue
    if os.path.exists(TLECAT_PATH):
        l0, l1, l2 = [], [], []
        with open(TLECAT_PATH, 'r') as f:
            for l, line in enumerate(f.readlines()):
        
                if line[-1] == '\n':
                    line = line[:-1]
        
                if l % 3 == 0:
                    l0.append(line)
                elif l % 3 == 1:
                    l1.append(line)
                else:
                    l2.append(line) 
        
        tle_cat = []
        for i in range(len(l0)):
            tle_cat.append(TLE(l1[i], l2[i], name=l0[i]))  
    else:
        print('Invalid path to TLE catalogue...')
        quit()
    
    print('TLEs in catalogue: {}'.format(len(tle_cat)))
    
    return sat_cat, tle_cat

def inSky(sat_cat, tle_cat, epoch, site, alt_limit):
    """
    Propagate TLE catalogue to desired epoch and check visibility
    
    Parameters
    ----------
    sat_cat : astropy.table.Table
        Table containing satcat information
    tle_cat : array-like
        List of TLEs [st.tle.TLE]
    epoch : datetime.datetime
        Epoch for TLE propagation
    site : st.site.Site
        Observation site for TLE propagation
    alt_limit : float
        Altitude limit for visibility check [degrees]
    
    Returns
    -------
    prop_table : astropy.table.Table
        Table of visible objects containing satcat and propagation information
    """
    # construct tle template and parse propagation info
    rolling_tle = TLE()
    rolling_tle.parse_propagation_info(epoch, site)
    
    # modify rolling tle and propagate to desired epoch
    propagation_log = Log(n_rows=len(tle_cat),
                          col_names=['NORAD_ID', 
                                     'NAME', 
                                     'TYPE',
                                     'COUNTRY', 
                                     'SIZE', 
                                     'LAUNCH_YR',
                                     'INCLINATION', 
                                     'HA', 
                                     'ALT',
                                     'FLAG'],
                          col_dtype=['i8', 'U25', 'U25', 
                                     'U25', 'U25', 'i8', 
                                     'f8', 'f8', 'f8', 'i8'])
    
    tle_cat_visible = []
    for t, tle in enumerate(tle_cat):
        rolling_tle.modify_elements(checksum=False, 
                                    norad_id=tle.norad_id.value, 
                                    designator={'yr': tle.designator.year, 
                                                'no': tle.designator.number, 
                                                'id': tle.designator.id}, 
                                    epoch=tle.epoch.date, 
                                    mmdot=tle.mmdot.value, 
                                    mmdot2=tle.mmdot2.value, 
                                    drag=tle.drag.value, 
                                    setnumber=tle.setnumber.value, 
                                    inclination=tle.inclination.value.deg, 
                                    raan=tle.raan.value.deg, 
                                    eccentricity=tle.eccentricity.value, 
                                    argperigee=tle.argperigee.value.deg, 
                                    meananomaly=tle.meananomaly.value.deg, 
                                    mm=tle.mm.value, 
                                    revnumber=tle.revnumber.value)
    
        ha = rolling_tle.propagate_ha()
        alt, _ = rolling_tle.propagate_altaz()
        
        # check visibility and log information
        if alt[0] > alt_limit:
            tle_cat_visible.append(tle)
            
            propagation_log.update('NORAD_ID', t, tle.norad_id.value)
            propagation_log.update('HA', t, ha[0].hourangle)
            propagation_log.update('ALT', t, alt[0])
        
            sat_cat_entry = sat_cat[sat_cat['NORAD_CAT_ID'] == tle.norad_id.value]
        
            if len(sat_cat_entry) == 1:
                propagation_log.update('NAME', t, tle.name)
                propagation_log.update('TYPE', t, sat_cat_entry['OBJECT_TYPE'][0])
                propagation_log.update('COUNTRY', t, sat_cat_entry['COUNTRY'][0])
                propagation_log.update('SIZE', t, sat_cat_entry['RCS_SIZE'][0])
                propagation_log.update('LAUNCH_YR', t, tle.designator.year)
                propagation_log.update('INCLINATION', t, tle.inclination.value.deg) 
            else:
                if len(sat_cat_entry) == 0:
                    propagation_log.update('FLAG', t, 2)  # missing
                else:
                    propagation_log.update('FLAG', t, 3)  # multiple      
        else:
            propagation_log.update('FLAG', t, 1)  # invisible
              
    return propagation_log.table[propagation_log.table['FLAG'] != 1], tle_cat_visible

def generateDummyTLE(epoch, site):
    """
    Generate dummy TLE for user selection
    
    Parameters
    ----------
    epoch : datetime.datetime
        Epoch for TLE propagation
    site : st.site.Site
        Observation site for TLE propagation
    
    Returns
    -------
    dummy_tle : st.tle.TLE
        Dummy (empty) TLE for user selection
    """
    day_secs = 24 * 60 * 60
    delta_t = [timedelta(seconds=i) for i in np.arange(-day_secs / 2, day_secs / 2, 100)]
    times = [epoch + dt for dt in delta_t]
    
    dummy_tle = TLE()
    dummy_tle.parse_propagation_info(times, site)
    
    return dummy_tle

def onPick(event):
    """
    Actions to perform upon picker event
    
    Parameters
    ----------
    event : Picker event
        User picked a point
    
    Returns
    -------
    None
    """
    global tle_cat, sat_cat, dummy_tle, prev_obj, fig, pick_ax, info_ax, show_ax
    
    # clear axes
    info_ax.cla()
    show_ax.cla()
    
    info_ax.axis('off')
    
    if prev_obj is not None:
        pick_ax.plot(prev_obj['HA'], prev_obj['ALT'], color='gray', marker='o')
    
    # identify selected object
    idx = event.ind[0]
    obj = sat_cat[idx] 
    
    pick_ax.plot(obj['HA'], obj['ALT'], 'ro')
    
    # update info axis
    info_ax.text(0.1, 0.8, 'Selected object', transform=info_ax.transAxes)
    for l, label in enumerate(LABELS):
        info_ax.text(0.1, 0.7 - l * 0.1, '{}:'.format(label[0]), transform=info_ax.transAxes)
        info_ax.text(0.4, 0.7 - l * 0.1, obj[label[1]], transform=info_ax.transAxes)
    
    # update show axis
    tle = tle_cat[idx]
    
    dummy_tle.modify_elements(checksum=False, 
                              norad_id=tle.norad_id.value, 
                              designator={'yr': tle.designator.year, 
                                          'no': tle.designator.number, 
                                          'id': tle.designator.id}, 
                              epoch=tle.epoch.date, 
                              mmdot=tle.mmdot.value, 
                              mmdot2=tle.mmdot2.value, 
                              drag=tle.drag.value, 
                              setnumber=tle.setnumber.value, 
                              inclination=tle.inclination.value.deg, 
                              raan=tle.raan.value.deg, 
                              eccentricity=tle.eccentricity.value, 
                              argperigee=tle.argperigee.value.deg, 
                              meananomaly=tle.meananomaly.value.deg, 
                              mm=tle.mm.value, 
                              revnumber=tle.revnumber.value)
    
    ha = dummy_tle.propagate_ha()
    alt, _ = dummy_tle.propagate_altaz()
    ra, dec = dummy_tle.propagate_radec()
    
    show_ax.plot(ha, alt, 'r.')
    
    show_ax.set_xlabel('Hour Angle [hrs]')
    show_ax.set_ylabel('Altitude [deg]')
    
    print('---------------------\n'
          'Selected object:\n'
          'Name:   {}\n'
          'ID:     {}\n'
          'Type:   {}\n'
          'Nation: {}\n'
          'Size:   {}\n'
          'Launch: {}\n'
          'Incl:   {}\n'
          '---------------------'.format(obj['NAME'],
                                         obj['NORAD_ID'],
                                         obj['TYPE'],
                                         obj['COUNTRY'],
                                         obj['SIZE'],
                                         obj['LAUNCH_YR'],
                                         obj['INCLINATION']))
    fig.canvas.draw()
    
    # store for next selection
    prev_obj = obj
    
    return None

def searchSky():
    """
    Run inMySky - interactive plot showing GSO objects visible in user's sky
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None
    """
    global tle_cat, sat_cat, dummy_tle, prev_obj, fig, pick_ax, info_ax, show_ax
    
    # parse user input arguments
    epoch, site = parseInput()
    
    # load satcat and tle catalogue
    sat_cat, tle_cat = loadCats(SATCAT_PATH, TLECAT_PATH)  # hard coded for testing
    
    # propagate tle catalogue to desired epoch and filter to obtain visible targets
    sat_cat, tle_cat = inSky(sat_cat, tle_cat, epoch, site, 10.)  
    print('Number of objects visible: {}'.format(len(sat_cat)))
    
    # prepare dummy tle for user selection
    dummy_tle = generateDummyTLE(epoch, site)
    prev_obj = None
    
    # set up interface and enable user interaction
    fig = plt.figure(figsize=(12,8))
    grid = plt.GridSpec(2, 3, wspace=0.05, hspace=0.05)

    pick_ax = fig.add_subplot(grid[:,:2])
    info_ax = fig.add_subplot(grid[0,2])
    show_ax = fig.add_subplot(grid[1,2])
    
    pick_ax.plot(sat_cat['HA'], sat_cat['ALT'], 'ko', picker=True, pickradius=5)
    
    pick_ax.axvline(x=0, color='r', linestyle='--')
    
    pick_ax.set_title('InYourSky | Epoch <{}> | {}'.format(epoch, site))
    
    pick_ax.set_xlabel('Hour Angle [hrs]')
    pick_ax.set_ylabel('Altitude [deg]')
    
    info_ax.axis('off')
    info_ax.set_title('Selected object:')
    
    show_ax.yaxis.tick_right()
    show_ax.yaxis.set_label_position('right')
    
    show_ax.set_xlabel('Hour Angle [hrs]')
    show_ax.set_ylabel('Altitude [deg]')
    
    fig.canvas.mpl_connect('pick_event', onPick)
    plt.show()
    
    return None
