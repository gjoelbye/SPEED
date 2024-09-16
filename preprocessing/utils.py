from collections import OrderedDict
import mne
import numpy as np

def _heuristic_eeg_resolution(eeg_ch_name: str):
    return eeg_ch_name if len(eeg_ch_name) > 0 else None

def _heuristic_eog_resolution(eog_channel_name):
    return eog_channel_name

def _heuristic_ref_resolution(ref_channel_name: str):
    if ref_channel_name.find('A1') != -1:
        return 'A1'
    elif ref_channel_name.find('A2') != -1:
        return 'A2'
    if ref_channel_name.find('L') != -1:
        return 'A1'
    elif ref_channel_name.find('R') != -1:
        return 'A2'
    else:
        return "REF"

def _heuristic_ecg_resolution(ecg_ch_name: str):
    # Get integer in string
    n = ''.join([i for i in ecg_ch_name if i.isdigit()])
    return "EKG" + n

def _heuristic_extra_resolution(extra_ch_name: str):
    return extra_ch_name

def _clean_channel_name(channel_name: str):
    channel_name = channel_name.upper()
    if channel_name == "REF": return channel_name
    channel_name = channel_name.replace('EEG', '')
    channel_name = channel_name.replace('LE', '')
    channel_name = channel_name.replace('REF', '')
    channel_name = channel_name.replace('EAR', '')
    channel_name = ''.join(c for c in channel_name if c.isalnum())
    return channel_name

def create_channel_type_dict(raw):
    type_dict = dict()
    for k, v in dict(zip(raw.info['ch_names'], raw.get_channel_types())).items():
        if any([x in k for x in ["A1", "A2"]]):
            type_dict[k] = 'ref'
        elif 'ECG' in k or 'EKG' in k:
            type_dict[k] = 'ecg'
        elif any([x in k for x in ['PHOTIC', 'IBI', 'BURSTS', 'SUPPR']]):
            type_dict[k] = 'extra'
        elif any([x in k for x in ['ROC', 'LOC']]):
            type_dict[k] = 'eog'  
        else:
            type_dict[k] = v

    return type_dict

def heuristic_resolution(old_type_dict: OrderedDict):
    resolver = {'eeg': _heuristic_eeg_resolution, 
                'eog': _heuristic_eog_resolution, 
                'ref': _heuristic_ref_resolution,
                'ecg': _heuristic_ecg_resolution,
                'extra': _heuristic_extra_resolution}
    
    new_type_dict = OrderedDict()
    
    for old_name, ch_type in old_type_dict.items():
        if ch_type is None:
            new_type_dict[old_name] = None
            continue
        
        new_name = _clean_channel_name(old_name)
        new_name = resolver[ch_type](new_name)
        
        if new_name is None:
            new_type_dict[new_name] = None
        else:
            while new_name in new_type_dict.keys():
                new_name = new_name + '-COPY'
            new_type_dict[new_name] = old_type_dict[old_name]

    assert len(new_type_dict) == len(old_type_dict)
    return new_type_dict

# ToDo: Transfer annotations
def split_raw(raw, window_length=60):
    sfreq = raw.info['sfreq']  # Sampling frequency
    windows_samples = int(window_length * sfreq)  # Samples per segment
    
    windows = []  # List to hold the segmented Raw objects
    time_slices = []  # List to hold the time slices for each segment
    start_sample = 0  # Initialize starting sample

    while start_sample < raw.n_times:
        end_sample = start_sample + windows_samples
        if end_sample > raw.n_times:
            break

        # Directly use the raw object's time slice method to get the segment
        # without copying the entire data set
        window, times = raw[:, start_sample:end_sample]
        
        # Creating a new RawArray object for each segment
        info = mne.create_info(ch_names=raw.info['ch_names'], sfreq=sfreq, ch_types=raw.get_channel_types())
        window_raw = mne.io.RawArray(window, info, verbose=False)
        window_raw.set_montage(raw.get_montage())
        
        windows.append(window_raw)
        time_slices.append((times[0], times[-1]))

        start_sample += windows_samples  # Move to the next segment

    return windows, time_slices


def get_unannotated_raw(raw, resting_state = ['T0']):
    # Initialize an empty list to hold the data of the non-annotated segments
    non_annotated_data_segments = []

    # The sample rate (number of samples per second)
    sfreq = raw.info['sfreq']

    # Convert annotation onset times from seconds to samples
    annotations = sorted(raw.annotations, key=lambda ann: ann['onset'])
    last_end_sample = 0

    for annotation in annotations:
        if annotation['description'] not in resting_state:
            onset_sample = int(annotation['onset'] * sfreq)
            end_sample = int((annotation['onset'] + annotation['duration']) * sfreq)
            
            if onset_sample > last_end_sample:
                # Extract non-annotated segment
                segment_data, _ = raw[:, last_end_sample:onset_sample]
                non_annotated_data_segments.append(segment_data)
                
            last_end_sample = end_sample

    # Don't forget to add the last segment if the last annotation does not reach the end of the data
    if last_end_sample < raw.n_times:
        segment_data, times = raw[:, last_end_sample:raw.n_times]
        non_annotated_data_segments.append(segment_data)

    # Concatenate all non-annotated data segments
    concatenated_data = np.concatenate(non_annotated_data_segments, axis=1)

    # Now, create a new RawArray with the concatenated non-annotated data
    info = mne.create_info(ch_names=raw.info['ch_names'], sfreq=sfreq, ch_types='eeg')
    non_annotated_raw = mne.io.RawArray(concatenated_data, info)
    return non_annotated_raw


def split_raw_annotations(raw, labels, tmin = -0.5, tlen = 5.0, verbose=True):
    windows = []
    time_slices = []
    descriptions = []

    for annotation in raw.annotations:
        onset, duration, description, _ = annotation.values()
        
        if description not in labels:
            continue
        
        sfreq = raw.info['sfreq']
        start_sample = round((onset + tmin) * sfreq)
        
        if start_sample < 0:
            if verbose: print(f'Skipping {description} at {onset:.2f} s')
            continue
        
        #durations_sample = round(duration * sfreq)
        tlen_sample = round(tlen * sfreq)    
        end_sample = start_sample + tlen_sample
        
        if end_sample > raw.n_times:
            if verbose: print(f'Skipping {description} at {onset:.2f} s')
            continue
        
        window, times = raw[:, start_sample:end_sample]
        
        info = mne.create_info(ch_names=raw.info['ch_names'], sfreq=sfreq, ch_types=raw.get_channel_types())
        window_raw = mne.io.RawArray(window, info, verbose=False)
        window_raw.set_montage(raw.get_montage())
        
        windows.append(window_raw)
        time_slices.append((times[0], times[-1]))
        descriptions.append(description)
        
    return windows, time_slices, descriptions

# Make custom montage for TUH dataset
def make_tuh_montage():
    # Load the standard_1005 montage
    standard_1005 = mne.channels.make_standard_montage('standard_1005')

    # Load the standard_postfixed montage to find T1 and T2 positions
    standard_postfixed = mne.channels.make_standard_montage('standard_postfixed')

    # Extract T1 and T2 positions from standard_postfixed
    t1_pos = standard_postfixed.get_positions()['ch_pos']['T1']
    t2_pos = standard_postfixed.get_positions()['ch_pos']['T2']

    # Combine standard_1005 positions with T1 and T2
    pos = standard_1005.get_positions()
    
    ch_pos = pos['ch_pos']
    ch_pos['T1'] = t1_pos
    ch_pos['T2'] = t2_pos
    
    ch_pos = OrderedDict({ch.lower().capitalize() : pos for ch, pos in ch_pos.items()})

    # Now create the new montage
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, nasion=pos['nasion'],
                                                lpa=pos['lpa'], rpa=pos['rpa'],
                                                hsp=pos['hsp'] , hpi=pos['hpi'], coord_frame=pos['coord_frame'])
    
    return montage