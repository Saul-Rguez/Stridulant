# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 10:30:39 2025
pipeline for the ai
@author: Saul
"""
# start by importing the module. I like to import as, you should be able to 
# import the way you want.

import stridulant as st

# the code will import its own libraries when needed, but I will be using some
# myself in this code, so I need to import them
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt

#let's simply add the path to the files to be processed
path = 'F:/estridulaciones/anteater data/tagged_files'


#%%
# I will first process all audios in a folder. You may process one by one 
# or extract features manually.
for file in os.listdir(path):
    if file.endswith('.wav'):
        st.process_audio_file(os.path.join(path,file))


#%%
#the function process_audio_file creates subfolders, this line captures them
# to make it work easily. Again, this can be done manually or implemented in 
# the code.
subdirectories = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

#now we will divide those into positives and negatives using the annotations files
# this block is a little complex, good news is that if you use the structures
# and naming conventions of stridulant as they are, you don't have to touch it.
# it will also create copies of all the data in merged folders for model training
# This code will likely become a feature of Stridulant, for now it is part of the
# pipeline, I'll see if I can implement this as a separate function or something
for file in os.listdir(path):
    if file.endswith('.txt'):
        notes =  os.path.join(path, file)
        for directory in subdirectories:
            if file[0:-4] in directory:
                snip = os.path.join(directory,"Audio_snippets")
                spec = os.path.join(directory,"Spectrograms")
                st.annotate_data(notes, snip, spec, snippet_duration=2.0, csv_delim = '\t')
            
            if not os.path.exists(os.path.join(path,"Merged_audio")):
                os.makedirs(os.path.join(path,"Merged_audio/Merged_positives"))     
                os.makedirs(os.path.join(path,"Merged_audio/Merged_negatives"))
            if not os.path.exists(os.path.join(path,"Merged_spectrograms")):
                os.makedirs(os.path.join(path,"Merged_spectrograms/Merged_spectrogram_positives"))
                os.makedirs(os.path.join(path,"Merged_spectrograms/Merged_spectrogram_negatives"))                             
                
            merged_audio_pos_path =os.path.join(path,"Merged_audio/Merged_positives")
            merged_audio_neg_path =os.path.join(path,"Merged_audio/Merged_negatives")
            merged_spec_pos_path =os.path.join(path,"Merged_spectrograms/Merged_spectrogram_positives")
            merged_spec_neg_path =os.path.join(path,"Merged_spectrograms/Merged_spectrogram_negatives")
        
            for file_name in os.listdir(os.path.join(snip,"positives")):
                source_path = os.path.join(snip,"positives", file_name)
                if os.path.isfile(source_path):
                    shutil.copy2(source_path, merged_audio_pos_path)
            
            for file_name in os.listdir(os.path.join(snip,"negatives")):
                source_path = os.path.join(snip,"negatives", file_name)
                if os.path.isfile(source_path):
                    shutil.copy2(source_path, merged_audio_neg_path)

            for file_name in os.listdir(os.path.join(spec,"positives")):
                source_path = os.path.join(spec,"positives", file_name)
                if os.path.isfile(source_path):
                    shutil.copy2(source_path, merged_spec_pos_path)
                    
            for file_name in os.listdir(os.path.join(spec,"negatives")):
                source_path = os.path.join(spec,"negatives", file_name)
                if os.path.isfile(source_path):
                    shutil.copy2(source_path, merged_spec_neg_path)
                    
#%%
#if the data is very biased (having more of one class than the other) like yours
# that has something like 350 negatives for each positive, there are 2 main 
# strategies to follow. First is data augmentation. you package has functions
#to do some personalized data augmentation in ways that make sense for sounds.
# this chunk of code will create variations of the stridulations and add them to
# the positives folder.


for file in os.listdir(merged_audio_pos_path):
    snippet = st.load_snippet(os.path.join(merged_audio_pos_path, file))
    st.create_variations(snippet, os.path.join(merged_audio_pos_path,'synthetic'),num_variations = 20)
    
for file in os.listdir(os.path.join(merged_audio_pos_path,'synthetic')):
    snippet = st.load_snippet(os.path.join(merged_audio_pos_path,'synthetic',file))
    spec=snippet.spectrogram('mel')
    spec.save(merged_spec_pos_path)
    
                   
                    
#%%                
# It's trraining time. With the apropriate function I will now run a training
# session. As it is, the model will automatically balance weights od the classes
# according to the amount of data on each.

mod, hist = st.train_model('F:/estridulaciones/anteater data/tagged_files/Merged_spectrograms', epochs = 10, class_weights = {0:1,1:50})
#%%

hist = pd.read_csv(os.path.join(path,'Merged_spectrograms/training_history.csv'))
plt.figure(figsize=(10, 6))

plt.plot(hist['loss'], label='training loss')
plt.plot(hist['val_loss'], label='validation loss')
plt.title('loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend()


plt.figure(figsize=(10, 6))
plt.plot(hist['accuracy'], label='training accuracy')
plt.plot(hist['val_accuracy'], label='validation accuracy')

plt.title('accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

#%%
spec_path = os.path.join(path,'test')
mod = ('F:/estridulaciones/anteater data/tagged_files/Merged_spectrograms/stridulation_detection_model.keras')
st.classify_spectrograms (spec_path,mod)









































        