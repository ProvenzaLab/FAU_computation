# extracting_FAUs

Extract Facial Actiona Units (FAUs) for multiple videos using OpenGraphAU (https://github.com/lingjivoo/OpenGraphAU) fork with pre-trained ResNet-18 weights.
Predict boolean detected face with OpenCV using Haar Casacades.

Installation using `uv venv`, then `uv sync`.

Wrapper call combined:
 - call `pipeline.py` with `PATH_VIDEO`

Or within individual steps:
 - Call `batch_run_whole.py`, pass PATH_VIDEO to predict FAUs
 - .npy file with frame-wise prediction will be written to same folder 
 - Call `filter_faces.py` to capture if face is present
 - Call `create_csv.py` to write out combined csv


```
AU names & indexes
{ 'AU_1': 'Inner brow raiser',  
  'AU_2': 'Outer brow raiser',  
  'AU_4': 'Brow lowerer',  
  'AU_5': 'Upper lid raiser',  
  'AU_6': 'Cheek raiser',  
  'AU_7': 'Lid tightener',  
  'AU_9': 'Nose wrinkler',  
  'AU_10': 'Upper lip raiser',  
  'AU_11': 'Nasolabial deepener',  
  'AU_12': 'Lip corner puller',  
  'AU_13': 'Sharp lip puller',  
  'AU_14': 'Dimpler',  
  'AU_15': 'Lip corner depressor',  
  'AU_16': 'Lower lip depressor',  
  'AU_17': 'Chin raiser',  
  'AU_18': 'Lip pucker',  
  'AU_19': 'Tongue show',  
  'AU_20': 'Lip stretcher',  
  'AU_22': 'Lip funneler',  
  'AU_23': 'Lip tightener',  
  'AU_24': 'Lip pressor',  
  'AU_25': 'Lips part',  
  'AU_26': 'Jaw drop',  
  'AU_27': 'Mouth stretch',  
  'AU_32': 'Lip bite',  
  'AU_38': 'Nostril dilator',  
  'AU_39': 'Nostril compressor',  
  'AU_L1': 'Left Inner brow raiser',  
  'AU_R1': 'Right Inner brow raiser',  
  'AU_L2': 'Left Outer brow raiser',  
  'AU_R2': 'Right Outer brow raiser',  
  'AU_L4': 'Left Brow lowerer',  
  'AU_R4': 'Right Brow lowerer',  
  'AU_L6': 'Left Cheek raiser',  
  'AU_R6': 'Right Cheek raiser',  
  'AU_L10': 'Left Upper lip raiser',  
  'AU_R10': 'Right Upper lip raiser',  
  'AU_L12': 'Left Nasolabial deepener*',  
  'AU_R12': 'Right Nasolabial deepener*',  
  'AU_L14': 'Left Dimpler',  
  'AU_R14': 'Right Dimpler',  
}
```

*OpenGraphAU* repo labels L/R Nasolabial deepener as AU12 (when it is usually AU11), the results reflect the opengraphAU labeling for consistency with the repo

Credits to [gbezold1](https://github.com/gbezold1) for setting this up!