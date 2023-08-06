#!/usr/bin/env python
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas  as pd
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import KNeighborsClassifier
import joblib


label_mapping = {'IoTAnalytics-TP-Link-Smart-plug_flows.csv': 1,
    'MoniotrPublic_google_home_mini_flows.csv': 2,
    'MoniotrPublic_cloudcam_flows.csv': 3,
    'MoniotrPublic_washer_flows.csv': 4,
    'MoniotrPublic_zmodo-doorbell_flows.csv': 5,
    'MoniotrPublic_philips-bulb_flows.csv': 6,
    'MoniotrPublic_microwave_flows.csv': 7,
    'MoniotrPublic_smartthings-hub_flows.csv': 8,
    'MoniotrPublic_microseven-camera_flows.csv': 9,
    'MoniotrPublic_dryer_flows.csv': 10,
    'MoniotrPublic_fridge_flows.csv': 11,
    'MoniotrPublic_ikettle_flows.csv': 12,
    'MoniotrPublic_xiaomi-ricecooker_flows.csv': 13,
    'MoniotrPublic_echospot_flows.csv': 14,
    'MoniotrPublic_magichome-strip_flows.csv': 15,
    'IoTAnalytics_Withings-Smart-Baby-Monitor_flows.csv': 16,
    'IoTAnalytics_HP-Printer_flows.csv': 17,
    'IoTAnalytics_Triby-Speaker_flows.csv': 18,
    'IoTAnalytics_iHome_flows.csv': 19,
    'IoTAnalytics_Amazon-Echo_flows.csv': 20,
    'google-home-mini_flows.csv': 2,
    'smartthings-hub_flows.csv': 8,
    'tplink-plug_flows.csv': 1,
    'echospot_flows.csv': 14,
    'cloudcam_flows.csv': 3,
    'fridge_flows.csv': 11,
    'magichome-strip_flows.csv': 15,
    'philips-bulb_flows.csv': 6,
    'washer_flows.csv': 4,
    'xiaomi-ricecooker_flows.csv': 13,
    'ikettle_flows.csv': 12,
    'zmodo-doorbell_flows.csv': 5,
    'dryer_flows.csv': 10,
    'microseven-camera_flows.csv': 9,
    'microwave_flows.csv': 7,
}

not_used_features = ['timeFirst', 'timeLast','dstPort', 'srcPort', 'dstIP', 
                      'dstMac', 'srcIP', 'srcMac', 'srcMac_dstMac_numP','ethVlanID','ipOptCpCl_Num',
                     'icmpBFTypH_TypL_Code', 'ip6OptHH_D', 'ip6OptCntHH_D']

hex_features = ['flowStat', 'tcpFStat', 'ipTOS', 'ipFlags', 'ethType',
                 'tcpStates', 'icmpStat', 'icmpTmGtw', 'macStat','tcpAnomaly', 
                'tcpFlags',  'tcpMPF', 'tcpMPTBF', 'tcpMPdssF', 'tcpOptions']


string_features = ['%dir', 'hdrDesc', 'srcManuf_dstManuf', 'dstPortClass', 
                   'srcIPCC', 'dstIPCC', 'dstIPOrg', 'srcIPOrg']

features = ['duration', 'numHdrDesc', 'numHdrs',
             'l4Proto', 'macPairs', 'dstPortClassN', 'numPktsSnt', 'numPktsRcvd',
            'numBytesSnt', 'numBytesRcvd', 'minPktSz', 'maxPktSz', 'avePktSize', 'stdPktSize', 'pktps', 'bytps',
            'pktAsm', 'bytAsm', 'ipMindIPID', 'ipMaxdIPID', 'ipMinTTL', 'ipMaxTTL', 'ipTTLChg', 'ipOptCnt',
            'tcpPSeqCnt', 'tcpSeqSntBytes', 'tcpSeqFaultCnt', 'tcpPAckCnt', 'tcpFlwLssAckRcvdBytes', 'tcpAckFaultCnt',
            'tcpInitWinSz', 'tcpAveWinSz', 'tcpMinWinSz', 'tcpMaxWinSz', 'tcpWinSzDwnCnt', 'tcpWinSzUpCnt',
            'tcpWinSzChgDirCnt', 'tcpOptPktCnt', 'tcpOptCnt', 'tcpMSS', 'tcpWS', 'tcpTmS', 'tcpTmER', 'tcpEcI',
            'tcpBtm', 'tcpSSASAATrip', 'tcpRTTAckTripMin', 'tcpRTTAckTripMax', 'tcpRTTAckTripAve',
            'tcpRTTAckTripJitAve', 'tcpRTTSseqAA', 'tcpRTTAckJitAve', 'icmpTCcnt', 'icmpEchoSuccRatio', 'icmpPFindex',
            'connSip', 'connDip', 'connSipDip', 'connSipDprt', 'connF',  'aveIAT', 'maxIAT', 
                'minIAT', 'stdIAT', 'tcpISeqN', 'tcpMPAID', 'tcpUtm', 'tcpWinSzThRt','label']

flows = pd.DataFrame()
for root, dirs, files in os.walk("flow_data/training"):
    for file in files:
        filepath = os.path.join(root, file)
        newFlow = pd.read_csv(filepath, index_col=None, header=0, delimiter='\t')
        newFlow["label"] = label_mapping[file]
        flows = pd.concat([flows, newFlow], axis=0, ignore_index=True)

flows = flows[features]

testing_flows = pd.DataFrame()
for root, dirs, files in os.walk("flow_data/testing"):
    for file in files:
        filepath = os.path.join(root, file)
        newFlow = pd.read_csv(filepath, index_col=None, header=0, delimiter='\t')
        newFlow["label"] = label_mapping[file]
        testing_flows = pd.concat([testing_flows, newFlow], axis=0, ignore_index=True)

testing_flows = testing_flows[features]

x = flows.iloc[:, flows.columns != 'label']
y = flows.iloc[:, flows.columns == 'label']

smote = SMOTE()
# fit predictor and target variable
x_smote, y_smote = smote.fit_resample(x, y)
x_smote['label'] = y_smote['label']
flows = x_smote

# Used for splitting and normalizing dataset.
#def test_scale()
X2_train = flows.iloc[:, flows.columns != 'label']
y2_train = flows.iloc[:, flows.columns == 'label']

X2_test = testing_flows.iloc[:, testing_flows.columns != 'label']
y2_test = testing_flows.iloc[:, testing_flows.columns == 'label']

#Normalizing
sc_X = StandardScaler()
X2_train = sc_X.fit_transform(X2_train)
X2_test = sc_X.transform(X2_test)

# X_train = X.iloc[:, X.columns != 'label']
# y_train = X.iloc[:, X.columns == 'label']

# X_test = Y.iloc[:, Y.columns != 'label']
# y_test = Y.iloc[:, Y.columns == 'label']

# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

# # Save StandardScaler
# joblib.dump(sc_X, 'std_scaler.bin', compress=True)

def generate_joblib(features_vetor):
    X = flows[features_vetor]
    Y = testing_flows[features_vetor]

    X_1 = X.iloc[:, X.columns != 'label']
    y = X.iloc[:, X.columns == 'label']

    X_train, X_test, y_train, y_test = train_test_split(X_1, y, test_size=0.50, random_state=1)

    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    # #### Random Forest Classifier
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train.values.ravel())

    y_pred = classifier.predict(X_test)
    print("\tRF Accuracy", round(metrics.accuracy_score(y_test, y_pred),2))

    ## save RandomForestClassifier model
    joblib.dump(classifier, "./random_forest.joblib")

    # #### K-Neighbors Classifier

    knn = KNeighborsClassifier(n_neighbors=20)
    knn.fit(X_train,  y_train.values.ravel())
    y_pred2 = knn.predict(X_test)
    print("\tKNN Accuracy", round(metrics.accuracy_score(y_test, y_pred2),2))

    joblib.dump(knn, "./KNeighborsClassifier.joblib")