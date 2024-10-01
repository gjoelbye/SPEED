import mne
import numpy as np
from mne.preprocessing import ICA
from mne_icalabel import label_components
import contextlib
import io
import pyprep
from meegkit.detrend import detrend
from meegkit.dss import dss_line_iter
from torch.nn.functional import interpolate
from torch import tensor
from src.utils import create_channel_type_dict, heuristic_resolution

class PreprocessMethods:
    def to_standard_names(raw):
        channel_type_dict = create_channel_type_dict(raw)
        channel_names = list(channel_type_dict.keys())
        revised_channel_types = heuristic_resolution(channel_type_dict)
        rename_map = {orig: new.lower().capitalize() for orig, new in zip(channel_names, revised_channel_types)}
        raw.rename_channels(rename_map)
        
        # Rename channels with same positions in standard 10-5 system
        rename_map = {'T3': 'T7', 'T4': 'T8', 'P7': 'T5', 'P8': 'T6'}
        rename_map = {k: v for k, v in rename_map.items() if k in raw.ch_names}
        
        raw.rename_channels(rename_map)

    def evaluate_quality(raw, oha_threshold=40e-6, thv_threshold=40e-6, chv_threshold=80e-6,
                        min_unique=100, min_nchans=10, line_freqs=[60], hp_freq=0.5, lp_freq=100):
        n_chans = raw.info['nchan']
        
        if n_chans < min_nchans:
            return False
        
        raw = raw.copy()
        
        # Discrete Channels
        unique_counts = np.array([len(np.unique(chan_data)) for chan_data in raw._data])
        discrete_channels = np.array(raw.ch_names)[unique_counts < min_unique].tolist()
        
        # Bad channels
        noisychannels = pyprep.NoisyChannels(raw)        
        noisychannels.find_bad_by_SNR()
        noisychannels.find_bad_by_correlation()
        noisychannels.find_bad_by_deviation()
        noisychannels.find_bad_by_hfnoise()
        noisychannels.find_bad_by_nan_flat()
        
        # Simple filtering
        raw.notch_filter(line_freqs, verbose=False)
        raw.filter(hp_freq, lp_freq, verbose=False)
        
        # Calculations
        oha = np.mean(np.abs(raw._data) > oha_threshold)
        thv = np.mean(np.std(raw._data, axis=0) > thv_threshold)
        chv = np.mean(np.std(raw._data, axis=1) > chv_threshold)

        bad_channels = discrete_channels + noisychannels.get_bads()
        bcr = len(bad_channels) / n_chans
        
        return (oha < 0.8) & (thv < 0.5) & (chv < 0.5) & (bcr < 0.8)     

    def interpolate_nearest(raw, sfreq=256.0):
        x = raw._data

        old_sfreq = raw.info['sfreq']
        
        resampled_data = interpolate(tensor(x).unsqueeze(0), 
                                     scale_factor=sfreq/old_sfreq, 
                                     mode="nearest").squeeze(0).numpy()
        
        lowpass = raw.info.get("lowpass")
        with raw.info._unlock():
            raw.info["sfreq"] = sfreq
            raw.info["lowpass"] = min(lowpass, sfreq / 2.0)
            
        raw._data = resampled_data
        raw._last_samps = np.array([resampled_data.shape[1] - 1])

    def set_montage(raw, montage):
        drop_chs = [ch for ch in raw.ch_names if ch not in montage.ch_names]
        raw.drop_channels(drop_chs)
        raw.set_montage(montage)
        return drop_chs
    
    def find_bad_channels(raw, ransac = True, drop = True):
        noisychannels = pyprep.NoisyChannels(raw)                
        noisychannels.find_bad_by_deviation(deviation_threshold=5.0)
        noisychannels.find_bad_by_hfnoise()
        noisychannels.find_bad_by_correlation(frac_bad=0.05)
        noisychannels.find_bad_by_SNR()
        
        try:
            if ransac and raw.info['nchan'] >= 16:
                noisychannels.find_bad_by_ransac()
        except Exception as e:
            pass
            #logging.error(f"Error in RANSAC: {e}")
        
        bad_chs = noisychannels.get_bads()
            
        # Drop bad channels
        if drop: raw.drop_channels(bad_chs)    
        
        return bad_chs

    def filter(raw, hp_freq, lp_freq, line_freqs, do_detrend=True):
        sfreq = raw.info['sfreq']
        nyquist = sfreq / 2.0
        
        # Blocks verbose output from MeegKit
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            if do_detrend:
                raw._data = detrend(raw._data.T, 1)[0].T # Might be anti-causal?
        
            for line_freq in line_freqs:
                if line_freq < nyquist:
                    if lp_freq is not None and line_freq < lp_freq:
                        continue
                    else:
                        raw._data = dss_line_iter(raw._data.T, line_freq, sfreq)[0].T
                    
        # Filtering
        raw.filter(hp_freq, lp_freq, verbose=False)
    
    def ica_clean(raw, iclabel_threshold, included_components):
        
        # Initialize ICA with the highest possible number of components
        # Infomax is important for ICLabel
        ica = ICA(
            n_components=np.linalg.matrix_rank(raw._data),
            max_iter=1000,
            method="infomax",
            fit_params=dict(extended=True),
            verbose=False
        )
        
        ica.fit(raw, verbose=False)

        # Run ICLabel
        ic_labels = label_components(raw, ica, method="iclabel")
        labels = ic_labels["labels"]
        y_proba = ic_labels["y_pred_proba"]
        
        # Exclude components not in the included components and with high probability
        exclude_idx = [
            idx for idx, (label, y_prob) in enumerate(zip(labels, y_proba))
            if label not in included_components and y_prob > iclabel_threshold
        ]

        # Recover the raw data
        raw = ica.apply(raw, exclude=exclude_idx, verbose=False)
        return exclude_idx, labels, y_proba
    
    def interpolate_missing(raw, chs, montage, mode="accurate"):
        missing_ch = [c for c in chs if c not in raw.ch_names]
        if len(missing_ch) == 0: return missing_ch
        
        # Adding placeholder missing channels
        new_channel_data = np.nan * np.zeros((len(missing_ch), raw._data.shape[1]))
        new_channel_info = mne.create_info(missing_ch, sfreq=raw.info['sfreq'], ch_types='eeg')
        raw.add_channels([mne.io.RawArray(new_channel_data, new_channel_info, verbose=False)], force_update_info=True)
        raw.info['bads'] = missing_ch
        
        # Setting montage for added channels
        raw.set_montage(montage, verbose=False)
        
        # Built-in intepolation
        raw.interpolate_bads(reset_bads=True, mode=mode, verbose=False)
        return missing_ch
    
    def zero_missing(raw, chs, montage):
        missing_ch = [c for c in chs if c not in raw.ch_names]
        if len(missing_ch) == 0: return missing_ch
        
        # Adding placeholder missing channels
        new_channel_data = np.zeros((len(missing_ch), raw._data.shape[1]))
        new_channel_info = mne.create_info(missing_ch, sfreq=raw.info['sfreq'], ch_types='eeg')
        raw.add_channels([mne.io.RawArray(new_channel_data, new_channel_info, verbose=False)], force_update_info=True)
        
        # Setting montage for added channels
        raw.set_montage(montage, verbose=False)
        return missing_ch
    
    def drop_extra_and_reorder(raw, chs):
        # List of extra channels
        extra_ch = [c for c in raw.ch_names if c not in chs]
        raw.drop_channels(extra_ch)
        
        # List of channel order
        new_ch_order = [ch for ch in chs if ch in raw.ch_names]    
        raw.reorder_channels(new_ch_order)    
        
        return extra_ch
    