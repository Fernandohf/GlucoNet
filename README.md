# GlucoNet
Network to predict glucose levels using RNN, GRU and LSTM.

Check this [link](https://fernandohf.github.io/GlucoNet/index.html) for detailed description.
___

## Performed Steps

### 1. Acquire te Data

Unfortunately, I don't have permission to share the data used. Therefore, to replicate the steps the [NightScout](http://www.nightscout.info/) data should be request at [Open Humans](https://www.openhumans.org/).

### 2. Data  Cleaning and Preparation
The file `data_extractor.py` has all the functions needed to prepare and clean the data **raw data**.

### 3. Define and Train the Models
The file `Data Analysis.ipynb` shows all the steps needed to feed the data in Recurrent Network Architecture. It goes from the **Exploratory Data Analysis (EDA)** to **Data Resample** and **Visualization**. Regarding the models, three different architectures of **Sequential Networks** were used:
- Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Units (GRU)

# Results
The models were used to predict the blood glucose during a period of 5 hours after a 10 hour prime period. The results are shown bellow.

![](https://i.imgur.com/nnP6ws1.png)

# Next Steps
Unfortunately, the data description and quality is lacking. Currently, I am searching for a more robust dataset which could result in more reliable and efficient models. Hopefully, reducing the prime period and increasing the predictions window.
In future, these models could be used to improved closed loop insulin systems.