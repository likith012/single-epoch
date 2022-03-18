import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import mne, os
from mne.datasets.sleep_physionet.age import fetch_data

from braindecode.datautil.preprocess import preprocess, Preprocessor
from braindecode.datautil.windowers import create_windows_from_events
from braindecode.datautil.preprocess import zscore
from braindecode.datasets import BaseConcatDataset, BaseDataset

from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from sklearn.utils import check_random_state

PATH = '/scratch/sleepkfold500single_epoch/'
DATA_PATH = '/scratch/'
os.makedirs(PATH, exist_ok=True)

# Params
BATCH_SIZE = 1
POS_MIN = 1
NEG_MIN = 15
EPOCH_LEN = 7
NUM_SAMPLES = 500
SUBJECTS = np.arange(83)
RECORDINGS = [1, 2]


##################################################################################################

random_state = 1234
n_jobs = 1
sfreq = 100
high_cut_hz = 30

window_size_s = 30
sfreq = 100
window_size_samples = window_size_s * sfreq



class SleepPhysionet(BaseConcatDataset):
    def __init__(
        self,
        subject_ids=None,
        recording_ids=None,
        preload=False,
        load_eeg_only=True,
        crop_wake_mins=30,
        crop=None,
    ):
        if subject_ids is None:
            subject_ids = range(83)
        if recording_ids is None:
            recording_ids = [1, 2]

        paths = fetch_data(
            subject_ids,
            recording=recording_ids,
            on_missing="warn",
            path= DATA_PATH,
        )

        all_base_ds = list()
        for p in paths:
            raw, desc = self._load_raw(
                p[0],
                p[1],
                preload=preload,
                load_eeg_only=load_eeg_only,
                crop_wake_mins=crop_wake_mins,
                crop=crop
            )
            base_ds = BaseDataset(raw, desc)
            all_base_ds.append(base_ds)
        super().__init__(all_base_ds)

    @staticmethod
    def _load_raw(
        raw_fname,
        ann_fname,
        preload,
        load_eeg_only=True,
        crop_wake_mins=False,
        crop=None,
    ):
        ch_mapping = {
            "EOG horizontal": "eog",
            "Resp oro-nasal": "misc",
            "EMG submental": "misc",
            "Temp rectal": "misc",
            "Event marker": "misc",
        }
        exclude = list(ch_mapping.keys()) if load_eeg_only else ()

        raw = mne.io.read_raw_edf(raw_fname, preload=preload, exclude=exclude)
        annots = mne.read_annotations(ann_fname)
        raw.set_annotations(annots, emit_warning=False)

        if crop_wake_mins > 0:
            # Find first and last sleep stages
            mask = [x[-1] in ["1", "2", "3", "4", "R"] for x in annots.description]
            sleep_event_inds = np.where(mask)[0]

            # Crop raw
            tmin = annots[int(sleep_event_inds[0])]["onset"] - crop_wake_mins * 60
            tmax = annots[int(sleep_event_inds[-1])]["onset"] + crop_wake_mins * 60
            raw.crop(tmin=max(tmin, raw.times[0]), tmax=min(tmax, raw.times[-1]))

        # Rename EEG channels
        ch_names = {i: i.replace("EEG ", "") for i in raw.ch_names if "EEG" in i}
        raw.rename_channels(ch_names)

        if not load_eeg_only:
            raw.set_channel_types(ch_mapping)

        if crop is not None:
            raw.crop(*crop)

        basename = os.path.basename(raw_fname)
        subj_nb = int(basename[3:5])
        sess_nb = int(basename[5])
        desc = pd.Series({"subject": subj_nb, "recording": sess_nb}, name="")

        return raw, desc


dataset = SleepPhysionet(
    subject_ids=SUBJECTS, recording_ids=RECORDINGS, crop_wake_mins=30
)


preprocessors = [
    Preprocessor(lambda x: x * 1e6),
    Preprocessor("filter", l_freq=None, h_freq=high_cut_hz, n_jobs=n_jobs),
]

# Transform the data
preprocess(dataset, preprocessors)


mapping = {  # We merge stages 3 and 4 following AASM standards.
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}

windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
    window_size_samples=window_size_samples,
    window_stride_samples=window_size_samples,
    preload= True,
    mapping=mapping,
)


preprocess(windows_dataset, [Preprocessor(zscore)])


###################################################################################################################################
""" Subject sampling """

rng = np.random.RandomState(1234)

NUM_WORKERS = 0 if n_jobs <= 1 else n_jobs
PERSIST = False if NUM_WORKERS <= 1 else True


subjects = np.unique(windows_dataset.description["subject"])
sub_pretext = rng.choice(subjects, 58, replace=False)
sub_test = sorted(list(set(subjects) - set(sub_pretext)))


print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(f"Pretext: {sub_pretext} \n")
print(f"Test: {sub_test} \n")
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


#######################################################################################################################################


class RelativePositioningDataset(BaseConcatDataset):
    """BaseConcatDataset with __getitem__ that expects 2 indices and a target."""

    def __init__(self, list_of_ds, epoch_len=7):
        super().__init__(list_of_ds)
        self.return_pair = True
        self.epoch_len = epoch_len

    def __getitem__(self, index):

        pos, neg = index
        pos_data = []
        neg_data = []

        assert pos != neg, "pos and neg should not be the same"

        for i in range(-(self.epoch_len // 2), self.epoch_len // 2 + 1):
            pos_data.append(super().__getitem__(pos + i)[0])
            neg_data.append(super().__getitem__(neg + i)[0])

        pos_data = np.stack(pos_data, axis=0) # (7, 2, 3000)
        neg_data = np.stack(neg_data, axis=0) # (7, 2, 3000)

        return pos_data, neg_data


class TuneDataset(BaseConcatDataset):
    """BaseConcatDataset for train and test"""

    def __init__(self, list_of_ds):
        super().__init__(list_of_ds)

    def __getitem__(self, index):

        X = super().__getitem__(index)[0]
        y = super().__getitem__(index)[1]

        return X, y


class RelativePositioningDataset(BaseConcatDataset):
    """BaseConcatDataset with __getitem__ that expects 2 indices and a target."""

    def __init__(self, list_of_ds, epoch_len=7):
        super().__init__(list_of_ds)
        self.return_pair = True
        self.epoch_len = epoch_len

    def __getitem__(self, index):

        pos, neg = index
        pos_data = []
        neg_data = []

        assert pos != neg, "pos and neg should not be the same"

        for i in range(-(self.epoch_len // 2), self.epoch_len // 2 + 1):
            pos_data.append(super().__getitem__(pos + i)[0])
            neg_data.append(super().__getitem__(neg + i)[0])

        pos_data = np.stack(pos_data, axis=0) # (7, 2, 3000)
        neg_data = np.stack(neg_data, axis=0) # (7, 2, 3000)

        return pos_data, neg_data


class TuneDataset(BaseConcatDataset):
    """BaseConcatDataset for train and test"""

    def __init__(self, list_of_ds):
        super().__init__(list_of_ds)

    def __getitem__(self, index):

        X = super().__getitem__(index)[0]
        y = super().__getitem__(index)[1]

        return X, y


class RecordingSampler(Sampler):
    def __init__(self, metadata, random_state=None, epoch_len=7):

        self.metadata = metadata
        self.epoch_len = epoch_len
        self._init_info()
        self.rng = check_random_state(random_state)

    def _init_info(self):
        keys = ["subject", "recording"]

        self.metadata = self.metadata.reset_index().rename(
            columns={"index": "window_index"}
        )
        self.info = (
            self.metadata.reset_index()
            .groupby(keys)[["index", "i_start_in_trial"]]
            .agg(["unique"])
        )
        self.info.columns = self.info.columns.get_level_values(0)

    def sample_recording(self):
        """Return a random recording index."""
        return self.rng.choice(self.n_recordings)
            
    def __iter__(self):
        raise NotImplementedError

    @property
    def n_recordings(self):
        return self.info.shape[0]


class RelativePositioningSampler(RecordingSampler):
    def __init__(
        self,
        metadata,
        tau_pos,
        tau_neg,
        n_examples,
        same_rec_neg=True,
        random_state=None,
        epoch_len=7,
    ):
        super().__init__(metadata, random_state=random_state, epoch_len=epoch_len)

        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.epoch_len = epoch_len
        self.n_examples = n_examples
        self.same_rec_neg = same_rec_neg
        self.info['index'] = self.info['index'].apply(lambda x: x[self.epoch_len // 2 : -(self.epoch_len // 2) ])
        self.info['i_start_in_trial'] = self.info['i_start_in_trial'].apply(lambda x: x[self.epoch_len // 2 : -(self.epoch_len // 2) ])
        self.info.iloc[-1]['index'] = self.info.iloc[-1]['index'][:-(7 // 2) - 1]
        self.info.iloc[-1]['i_start_in_trial'] = self.info.iloc[-1]['i_start_in_trial'][: -(self.epoch_len // 2) - 1]

    def _sample_pair(self):
        
        """Sample a pair of two windows."""
        # Sample first window
        
        for rec_id in range(self.info.shape[0]):
            epochs = self.info.iloc[rec_id]["index"]
            start_trail = self.info.iloc[rec_id]["i_start_in_trial"]
            for ep_id, trail in zip(epochs, start_trail):
               
                win_ind1, rec_ind1 = ep_id, rec_id
                ts1 = trail
                ts = self.info.iloc[rec_ind1]["i_start_in_trial"]

                epoch_min = self.info.iloc[rec_ind1]["i_start_in_trial"][self.epoch_len // 2]
                epoch_max = self.info.iloc[rec_ind1]["i_start_in_trial"][-self.epoch_len // 2]

                if self.same_rec_neg:
                    mask = ((ts <= ts1 - self.tau_neg) & (ts >= epoch_min)) | (
                        (ts >= ts1 + self.tau_neg) & (ts <= epoch_max)
                    )
                    
                if sum(mask) == 0:
                    raise NotImplementedError
                win_ind2 = self.rng.choice(self.info.iloc[rec_ind1]["index"][mask])
                yield win_ind1, win_ind2

    def __iter__(self):  
        yield from self._sample_pair()
      
    def __len__(self):
        epoch_len = 0
        for rec_id in range(self.info.shape[0]):
            epoch_len += len(self.info.iloc[rec_id]["index"])
        return epoch_len
    
    
######################################################################################################################


PRETEXT_PATH = os.path.join(PATH, "pretext")
TEST_PATH = os.path.join(PATH, "test")

if not os.path.exists(PRETEXT_PATH): os.mkdir(PRETEXT_PATH)
if not os.path.exists(TEST_PATH): os.mkdir(TEST_PATH)


splitted = dict()

splitted["pretext"] = RelativePositioningDataset(
    [ds for ds in windows_dataset.datasets if ds.description["subject"] in sub_pretext],
    epoch_len = EPOCH_LEN
)


splitted["test"] = [ds for ds in windows_dataset.datasets if ds.description["subject"] in sub_test]

for sub in splitted["test"]:
    temp_path = os.path.join(TEST_PATH, str(sub.description["subject"]) + str(sub.description["recording"])+'.npz')
    np.savez(temp_path, **sub.__dict__)

########################################################################################################################


# Sampler
tau_pos, tau_neg = int(sfreq * POS_MIN * 60), int(sfreq * NEG_MIN * 60)
n_examples_pretext = NUM_SAMPLES * len(splitted["pretext"].datasets)

print(f'Number of pretext subjects: {len(splitted["pretext"].datasets)}')
print(f'Number of pretext epochs: {n_examples_pretext}')

pretext_sampler = RelativePositioningSampler(
    splitted["pretext"].get_metadata(),
    tau_pos=tau_pos,
    tau_neg=tau_neg,
    n_examples=n_examples_pretext,
    same_rec_neg=True,
    random_state=random_state  # Same samples for every iteration of dataloader
)


# Dataloader
pretext_loader = DataLoader(
    splitted["pretext"],
    batch_size=BATCH_SIZE,
    sampler=pretext_sampler
)


for i, arr in tqdm(enumerate(pretext_loader), desc = 'pretext'):
    temp_path = os.path.join(PRETEXT_PATH, str(i) + '.npz')
    np.savez(temp_path, pos = arr[0].numpy().squeeze(0), neg = arr[1].numpy().squeeze(0))
  

