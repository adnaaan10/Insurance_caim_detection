import streamlit as st
import pandas as pd
import joblib


Model=joblib.load(open("xgb_model.pkl","rb"))
le=joblib.load(open("Labelencoder.pkl","rb"))
te=joblib.load(open("Targetencoder.pkl","rb"))
sc=joblib.load(open("Scaler.pkl","rb"))

st.title("Insurance Claim Prediction")

def Input():
    name=st.text_input('Enter Your Name:')
    customer_age=st.number_input("Enter Your Age:",min_value=18,value=None,step=1,format="%d")
    vehicle_name=st.text_input("Enter Your Vehicle Name:")
    subscription_length=st.number_input("Subscription_length:",value=None)
    vehicle_age=st.number_input("Enter Vehicle Age:",value=None)
    region_density=st.number_input("Enter Region Density:",value=None,step=1,format="%d")
    cylinder=st.number_input("Enter Cylinder Number:",value=None,step=1,format="%d")
    region_code=st.text_input("Enter Region Code:")
    model=st.text_input("Enter Your Vehicle Model:")
    max_torque=st.number_input("Enter Maximum Torque:",value=None)
    max_power=st.number_input("Enter Maximum Power:",value=None)
    is_adjustable_steering=st.selectbox("Steering Adjustable or Not",options=["Yes","No"])
    steering_type=st.text_input("Steering Type:")


    if st.button("submit"):
        if not all([name.strip(),
                    vehicle_name.strip(),
                    subscription_length,
                    vehicle_age,
                    customer_age,
                    region_density,
                    cylinder,
                    region_code.strip(),
                    model.strip(),
                    max_torque,
                    max_power,
                    is_adjustable_steering,
                    steering_type.strip()]):
            st.warning("All Fields are Mandatory")
            return None
        else:
            input_data=pd.DataFrame([[region_code, model, steering_type]],columns=['region_code', 'model', 'steering_type'])
            encoded_data=te.transform(input_data)
            LabelEncoderData=le.transform([is_adjustable_steering])
            data=pd.DataFrame([[subscription_length,vehicle_age,customer_age,region_density,cylinder,max_torque,max_power]],columns=['subscription_length', 'vehicle_age', 'customer_age', 'region_density','cylinder', 'max_torque', 'max_power'])
            data["is_adjustable_steering"]=LabelEncoderData
            data=pd.concat([data,encoded_data],axis=1)
            data=sc.transform(data)
            predicted_value=Model.predict(data)
            if predicted_value==1:
                st.success("your claim being approved ✅")
            else:
                st.error("Sorry Your Claim  Not Approved😢😢😢")
    else:
        return None













Input()