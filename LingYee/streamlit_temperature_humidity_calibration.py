import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.header("Temparature and humidity calibration")
st.image("tempchart.png")

# define a function for processing and cleaning the data
def data_proc(dataf):
    # drop duplicates so that the dataframe can be pivoted
    dataf = dataf.drop_duplicates()
    dataf = dataf.dropna()

    # pivot the data table
    dfpivot = dataf.pivot(index=["time", "source_address"], columns="parameter_name", values="parameter_value").reset_index()

    # change the type of "time" to datetime
    dfpivot["time"] = pd.to_datetime(dfpivot["time"])

    # add column on energy release or each period (take the difference between total of current period and total of previous period)
    df = dfpivot.copy()
    df["delta_energy"] = dfpivot.sort_values(["source_address", "time"], ascending=True).groupby("source_address")["total_normalized_energy"].diff()
    
    # add a column on the difference between current timestamp and previous timestamp (to find the time gap between readings)
    df["time_diff"] = dfpivot.sort_values(["source_address", "time"], ascending=True).groupby("source_address")["time"].diff().astype("timedelta64[s]")
    df = df[df["time_diff"]!=0]
    # add a column on the power release for each period
    df["power"] = (np.abs(df["delta_energy"]) * 21.5) / (df["time_diff"]) + 1.5

    # keep only the features needed by the model
    dflamp = df[["sht40_temperature", "sht40_humidity", "power"]]
    dflamp.rename(columns={"sht40_temperature": "lamp_temperature", "sht40_humidity": "lamp_humidity"})

    # drop missing values and reset index
    dflamp = dflamp.dropna().reset_index(drop=True)

    return dflamp

# define a function to load the saved scaler and model, then make prediction
def make_predict(scaler, model, dataf):
    # load the scaler and model from disk
    loaded_scaler = pickle.load(open(scaler, "rb"))
    loaded_model = pickle.load(open(model, "rb"))

    # standardize the dataset using loaded scaler
    dataf_scaled = loaded_scaler.transform(dataf) 

    # run the prediction using loaded model
    l_pred = loaded_model.predict(dataf_scaled)

    return l_pred

# define a function to print the table of output
def print_output(dataf, pred):
    df_out = pd.concat([dataf, pd.DataFrame(pred)], axis=1)
    if choice == "Temperature":
        df_out = df_out.rename(columns={"sht40_temperature": "lamp_temperature", "sht40_humidity": "lamp_humidity", 0: "predicted_temperature"})
        df_out.to_csv("temperature_prediction.csv", index=False)
    if choice == "Humidity":
        df_out = df_out.rename(columns={"sht40_temperature": "lamp_temperature", "sht40_humidity": "lamp_humidity", 0: "predicted_humidity"})
        df_out.to_csv("humidity_prediction.csv", index=False)

    # print the output table
    st.write("Table with input values and prediction:")
    st.dataframe(df_out)

    # provide the option to download the output table as a csv file
    if choice == "Temperature":
        with open("temperature_prediction.csv", "rb") as f: 
            st.download_button("Download prediction", f, file_name="temperature_prediction.csv")
    if choice == "Humidity":
        with open("humidity_prediction.csv", "rb") as f: 
            st.download_button("Download prediction", f, file_name="humidity_prediction.csv")

with st.sidebar: 
    st.info("This application helps to calibrate the room temperature and humidity based on reading from the sensor of the lamp.")
    choice = st.radio("Feature to calibrate: ", ["Temperature", "Humidity"])
    modelchoice = st.radio("Prediction model: ", ["Linear Regression", "Random Forest"])
    st.subheader("Single Lamp")
    lamp_temp = st.number_input("Enter the lamp temperature: ")
    lamp_humid = st.number_input("Enter the lamp humidity: ")
    lamp_power = st.number_input("Enter the power: ")
    predict = st.button("Show prediction")
    st.subheader("Multiple Lamps")
    file = st.file_uploader("Upload a raw csv file converted from SQL")

###################################################################################################
# ONLY NEED TO CHANGE THE FILE PATH IN THIS SECTION IF WANT TO CHANGE THE MODEL USED IN PREDICTION
###################################################################################################
# load the scaler and model from disk
if choice == "Temperature" and modelchoice == "Linear Regression":
    scalerfile = "saved_scaler_t.sav"
    modelfile = "saved_model_t.sav"
if choice == "Humidity" and modelchoice == "Linear Regression":
    scalerfile = "saved_scaler_h.sav"
    modelfile = "saved_model_h.sav"
if choice == "Temperature" and modelchoice == "Random Forest":
    scalerfile = "saved_scaler_t.sav"
    modelfile = "saved_modelrf_t.sav"
if choice == "Humidity" and modelchoice == "Random Forest":
    scalerfile = "saved_scaler_h.sav"
    modelfile = "saved_modelrf_h.sav"
###################################################################################################

if predict:
    # specify the dataset
    features = {"lamp_temperature": lamp_temp, "lamp_humidity": lamp_humid, "power": lamp_power }
    features_df  = pd.DataFrame([features])
    st.write("Input values:")
    st.table(features_df)

    # make prediction
    if modelchoice == "Linear Regression":
        prediction =  make_predict(scalerfile, modelfile, features_df)[0][0]
    if modelchoice == "Random Forest":
        prediction =  make_predict(scalerfile, modelfile, features_df)[0]
    if choice == "Temperature":
        st.metric(label="Based on the input values, the predicted room temperature is ", value=f"{round(prediction, 2)} °C")
    if choice == "Humidity":
        st.metric(label="Based on the input values, the predicted room humidity is ", value=f"{round(prediction, 2)} %")

if file and not predict:
    # specify the dataset
    df = pd.read_csv(file, index_col=None)
    
    # type 1 of csv file - if the file already contains the three main features
    try:
        # keep only the features needed by the model
        dflamp = df[["sht40_temperature", "sht40_humidity", "power"]]
        dflamp.rename(columns={"sht40_temperature": "lamp_temperature", "sht40_humidity": "lamp_humidity"})
        # drop missing values and reset index
        dflamp = dflamp.dropna().reset_index(drop=True)
    
    # type 2 of csv file - if the file is a raw file directly converted from the SQL file 
    except Exception:
        try:
            # process and clean the data if it is a raw file
            dflamp = data_proc(df)
        except Exception:
            st.write("DATA ERROR!!! The csv file should be comma-delimited and should contain the following columns: lamp_temperature, lamp_humidity, power")
        else:
            prediction = make_predict(scalerfile, modelfile, dflamp)
            print_output(dflamp, prediction)

    else:
        prediction = make_predict(scalerfile, modelfile, dflamp)
        print_output(dflamp, prediction)