from ml_classification import predict
from Generate_jolibs import generate_joblib
import pandas as pd
import os
from itertools import combinations

label_mapping = {2: 'google-home-mini_flows.csv',
    8: 'smartthings-hub_flows.csv',
    1: 'tplink-plug_flows.csv',
    14: 'echospot_flows.csv',
    3: 'cloudcam_flows.csv',
    11: 'fridge_flows.csv',
    15: 'magichome-strip_flows.csv',
    6: 'philips-bulb_flows.csv',
    4: 'washer_flows.csv',
    13: 'xiaomi-ricecooker_flows.csv',
    12: 'ikettle_flows.csv',
    5: 'zmodo-doorbell_flows.csv',
    10: 'dryer_flows.csv',
    9: 'microseven-camera_flows.csv',
    7: 'microwave_flows.csv',
}

features_no_reduce = ['duration', 'numHdrDesc', 'numHdrs',
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

features_reduce_zeros = ['duration', 'l4Proto','minPktSz','pktAsm','bytAsm','ipMindIPID','ipTTLChg','ipOptCnt','tcpPSeqCnt','tcpPAckCnt','tcpAckFaultCnt','tcpWinSzDwnCnt','tcpWinSzChgDirCnt','tcpOptPktCnt','tcpOptCnt','tcpWS','tcpSSASAATrip','tcpRTTAckTripMin','tcpRTTAckTripMax','tcpRTTAckTripAve','tcpRTTAckTripJitAve','tcpRTTSseqAA','tcpRTTAckJitAve','icmpTCcnt','icmpEchoSuccRatio','aveIAT','maxIAT','stdIAT','tcpUtm']

def evaluate_files(features):
    for root, dirs, files in os.walk("flow_data/testing"):
        for file in files:
            path = os.path.join(root, file)
            flow = pd.read_csv(path, delimiter='\t')
            flow = flow[features]
            rf, knn = predict(flow)

            rfScore = knnScore = None
            for i in rf:
                if i[0] in label_mapping.keys():
                    if file == label_mapping[i[0]]:
                        rfScore = i[1]
                        break
            for i in knn:
                if i[0] in label_mapping.keys():
                    if file == label_mapping[i[0]]:
                        knnScore = i[1]
                        break
            print(f"{file} \n\tRF: {rfScore} \n\tKNN:{knnScore}")

def brute_feature_eval():
    for array in combinations(features_reduce_zeros, 6):
        rf, knn = generate_joblib(list(array) + ["label"])
        with open("brute.csv", "a") as f:
            f.write(f"{list(array)},  {[x for x in features_reduce_zeros if x not in list(array)]},{rf}, {knn}\n")


if __name__ == "__main__":
    #features = ["connDip","connSip","minPktSz","duration","pktAsm","tcpRTTAckTripMax", "label"]
    #generate_joblib(features)
    #features.remove("label")
    #evaluate_files(features)``
    brute_feature_eval()