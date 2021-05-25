import streamlit as st
import accuracy
from trainmodel import get_wav, to_mfcc, create_segmented_mfccs, segment_one
from tensorflow import keras


st.title("This is a quick demo of The Accent Recognition App")

st.text("Audio samples to be tested")

st.text("American Accent")
american_filename = 'english33'
american = st.audio(f'./{american_filename}.wav')
# american = st.audio(f'./english33.wav')
# st.write(american)

st.text("British Accent")
british_filename = 'english38'
british = st.audio(f'./{british_filename}.wav')
# st.write(british)

st.text("Australian Accent")
australian_filename = 'english77'
australian = st.audio(f'./{australian_filename}.wav')
# st.write(australian)


st.subheader("Please select a test file from below to run with the accent app..")
option = st.selectbox('Select', ["American",'British','Australian'])
st.write('You Selected: ', option)
st.text(option)

option_dict = {"American":american_filename,'British':british_filename,'Australian':australian_filename}
keys_list = list(option_dict)


model = keras.models.load_model('./baseline_english.h5.h5')
if st.button('Start Testing'):
    test_load_state = st.text('Test in progress')
    wav_file = get_wav(option_dict[option])
    st.text(wav_file)
    mfcc = to_mfcc(wav_file)
    st.text(mfcc)
    y_predicted = accuracy.predict_class_audio(segment_one(mfcc), model)
    test_load_state.text(f'Testing done... the predicted accent is {keys_list[y_predicted-1]}')
    # st.write(test_load_state)
else:
    st.write('Test not yet started')