#=============================================================================
# Refinitiv Data Platform demo app to get timeSeries data
#-----------------------------------------------------------------------------
#   This source code is provided under the Apache 2.0 license
#   and is provided AS IS with no warranty or guarantee of fit for purpose.
#   Copyright (C) 2021 Refinitiv. All rights reserved.
#=============================================================================

import pandas as pd
import requests
import json
import rdpToken

# Application Constants
RDP_version = "/v1"
base_URL = "https://api.refinitiv.com"
category_URL = "/data/historical-pricing"
endpoint_URL = "/views/interday-summaries"
universe_parameter_URL = "/"

# Elenco dei RIC delle aziende dell'S&P 500
sp500_ric_list = [
    "AFL.N","AES.N","ABT.N","ADBE.OQ","AMD.OQ","APD.N","ALB.N","HON.OQ","ALL.N","HWM.N","HES.N","AEE.N","AEP.OQ",
    "AXP.N","AIG.N","COR.N","AME.N","AMGN.OQ","APH.N","ADI.OQ","AON.N","APA.OQ","AAPL.OQ","AMAT.OQ","ADM.N","ATO.N",
    "ADSK.OQ","ADP.OQ","AZO.N","AVB.N","AVY.N","TFC.N","BKR.OQ","BALL.N","BAX.N","BDX.N","VZ.N","WRB.N","BRKb.N",
    "BBY.N","BIO.N","BA.N","BWA.N","BSX.N","BMY.N","BFb.N","CI.N","CMS.N","CSX.OQ","CVS.N","CTRA.N","CDNS.OQ","CPT.N",
    "CPB.OQ","STZ.N","COF.N","CAH.N","CCL.N","CAT.N","JPM.N","CVX.N","CHD.N","CINF.OQ","CSCO.OQ","CTAS.OQ","CLX.N",
    "KO.N","CL.N","CAG.N","ED.N","COO.OQ","TAP.N","CPRT.OQ","GLW.N","CMI.N","DHI.N","DTE.N","DHR.N","DRI.N","TGT.N",
    "DECK.N","DE.N","DIS.N","DLTR.OQ","D.N","DOV.N","DUK.N","EMN.N","ETN.N","ECL.N","EIX.N","EA.OQ","EMR.N","EOG.N",
    "ETR.N","EFX.N","EQT.N","EQR.N","EG.N","EXPD.N","XOM.N","FMC.N","NEE.N","FICO.N","FAST.OQ","FDX.N","FRT.N","FITB.OQ",
    "FI.N","FE.N","BEN.N","FCX.N","AJG.N","IT.N","GD.N","GE.N","GIS.N","GPC.N","GILD.OQ","GWW.N","HAL.N","LHX.N","HIG.N",
    "HAS.OQ","DOC.N","WELL.N","JKHY.OQ","HSY.N","HPQ.N","HOLX.OQ","HD.N","HRL.N","HST.OQ","HUBB.N","HUM.N","JBHT.OQ",
    "HBAN.OQ","BIIB.OQ","MOS.N","IEX.N","IDXX.OQ","ITW.N","INCY.OQ","TT.N","INTC.OQ","IBM.N","IFF.N","IP.N","IPG.N",
    "INTU.OQ","JBL.N","J.N","JNJ.N","KLAC.OQ","K.N","KEY.N","KMB.N","KIM.N","KR.N","LH.N","LRCX.OQ","EL.N","LEN.N",
    "LLY.N","BBWI.N","LMT.N","L.N","LOW.N","MTB.N","MGM.N","MMC.N","MAR.OQ","MLM.N","MAS.N","MKC.N","MCD.N","SPGI.N",
    "MCK.N","MDT.N","BK.N","MSFT.OQ","MCHP.OQ","MU.OQ","MAA.N","MMM.N","MHK.N","MS.N","MSI.N","VTRS.OQ","NVR.N","NTAP.OQ",
    "NEM.N","NKE.N","NDSN.OQ","NSC.N","ES.N","XEL.OQ","NTRS.OQ","NOC.N","WFC.N","NUE.N","OXY.N","ODFL.OQ","OMC.N","OKE.N",
    "ORCL.N","ORLY.OQ","EXC.OQ","PCG.N","PNC.N","PPL.N","PPG.N","PCAR.OQ","PTC.OQ","PH.N","PAYX.OQ","PNR.N","PEP.OQ",
    "PFE.N","MO.N","COP.N","PNW.N","TROW.OQ","PG.N","PGR.N","PEG.N","PSA.N","PHM.N","QCOM.OQ","PWR.N","RJF.N","O.N",
    "REGN.OQ","REG.OQ","RMD.N","ACGL.OQ","ROK.N","ROL.N","ROP.OQ","ROST.OQ","T.N","TRV.N","HSIC.OQ","SLB.N","SCHW.N",
    "SRE.N","SHW.N","SPG.N","AOS.N","SNA.N","SO.N","LUV.N","SWK.N","USB.N","SBUX.OQ","STT.N","SYK.N","GEN.OQ","SNPS.OQ",
    "SYY.N","TJX.N","TECH.OQ","TFX.N","TER.OQ","TXN.OQ","TXT.N","TMO.N","GL.N","DVA.N","TSCO.OQ","C.N","YUM.N","TRMB.OQ",
    "TSN.N","MRO.N","WM.N","UNP.N","UDR.N","UNH.N","RTX.N","UHS.N","VLO.N","VTR.N","VRSN.OQ","VRTX.OQ","VMC.N","WMT.N",
    "WBA.OQ","WAT.N","WDC.OQ","EVRG.OQ","WAB.N","WY.N","WMB.N","WEC.N","ZBRA.OQ","CB.N","JCI.N","RCL.N","AMT.N","MCO.N",
    "DGX.N","STLD.OQ","FDS.N","PLD.N","URI.N","CNP.N","BXP.N","ESS.N","ARE.N","CHRW.OQ","MTD.N","IRM.N","WST.N","NI.N",
    "SWKS.OQ","BAC.N","AMZN.OQ","BRO.N","RL.N","LNT.OQ","TYL.N","CTSH.OQ","CCI.N","EBAY.OQ","GS.N","NVDA.OQ","BKNG.OQ",
    "RSG.N","POOL.OQ","CSGP.OQ","COST.OQ","DVN.N","RVTY.N","TTWO.OQ","AKAM.OQ","TDY.N","UPS.N","JNPR.N","EW.N","A.N",
    "BLK.N","FFIV.OQ","MET.N","PKG.N","SBAC.OQ","ON.OQ","F.N","GPN.N","TPR.N","ALGN.OQ","ANSS.OQ","CRL.N","KMX.N","ISRG.OQ",
    "FIS.N","ZBH.N","ACN.N","ELV.N","IVZ.N","BG.N","EQIX.OQ","GRMN.N","MNST.OQ","MDLZ.OQ","PFG.OQ","AXON.OQ","WTW.OQ","CNC.N",
    "PRU.N","NFLX.OQ","SJM.N","NDAQ.OQ","WYNN.OQ","CMCSA.OQ","CME.OQ","STX.OQ","MOH.N","LKQ.OQ","NRG.N","AIZ.N","CBRE.N","CRM.N",
    "RF.N","DPZ.N","TMUS.OQ","EXR.N","GOOGL.OQ","DLR.N","MKTX.OQ","MPWR.OQ","LVS.N","CE.N","DXCM.OQ","BLDR.N","EXPE.OQ","CF.N","AMP.N","MA.N",
    "ICE.N","LDOS.N","PARA.OQ","LYV.N","CMG.N","UAL.OQ","TDG.N","FSLR.OQ","KKR.N","BR.N","DAL.N","PODD.OQ","TEL.N","DFS.N","BX.N","LULU.OQ",
    "ULTA.OQ","MSCI.N","PM.N","V.N","AWK.N","KDP.OQ","WBD.OQ","AVGO.OQ","VRSK.OQ","MRK.N","DG.N","FTNT.OQ","CHTR.OQ","GNRC.N","LYB.N","CBOE.Z",
    "TSLA.OQ","NXPI.OQ","TRGP.N","KMI.N","HCA.N","GM.N","HII.N","MPC.N","XYL.N","APTV.N","EPAM.N","ENPH.OQ","CPAY.N","NCLH.N","META.OQ","FANG.OQ",
    "NOW.N","PSX.N","PANW.OQ","ABBV.N","ZTS.N","IQV.N","NWSA.OQ","NWS.OQ","CDW.OQ","ALLE.N","HLT.N","AAL.OQ","GOOG.OQ","ANET.N","PAYC.N","CTLT.N",
    "SYF.N","CFG.N","CZR.OQ","KEYS.N","QRVO.OQ","GDDY.N","ETSY.OQ","KHC.OQ","PYPL.OQ","HPE.N","MTCH.OQ","STE.N","FTV.N","LW.N","INVH.N","IR.N",
    "VST.N","DD.N","VICI.N","DAY.N","LIN.OQ","MRNA.OQ","FOXA.OQ","FOX.OQ","DOW.N","UBER.N","CRWD.OQ","CTVA.N","AMCR.N","SMCI.OQ","CARR.N","OTIS.N",
    "ABNB.OQ","CEG.OQ","GEHC.OQ","KVUE.N","VLTO.N","GEV.N","SOLV.N","SW.N"
]

def prettyPrintData(vData):
    print(json.dumps(vData, indent=2))

    line = ""
    for i in vData["headers"]:
        line = line + i["name"] + ", "
    line = line[:-2]
    print(line)

    print("---------------")

    for d in vData["data"]:
        line = ""
        for pt in d:
            line = line + str(pt) + ", "
        line = line[:-2]
        print(line)

def get_data_for_ric(accessToken, ric):
    RESOURCE_ENDPOINT = base_URL + category_URL + RDP_version + endpoint_URL + universe_parameter_URL + ric

    requestData = {
        "interval": "P1D",  # Intervallo giornaliero
        "start": "2013-01-01",  # Data di inizio
        "end": "2022-12-31",  # Data di fine
        "adjustments": "exchangeCorrection,manualCorrection,CCH,CRE,RPO,RTS",
        "fields": "BID,ASK,OPEN_PRC,HIGH_1,LOW_1,TRDPRC_1,NUM_MOVES,TRNOVR_UNS"
    }

    dResp = requests.get(RESOURCE_ENDPOINT, headers={"Authorization": "Bearer " + accessToken}, params=requestData)

    if dResp.status_code != 200:
        print(f"Unable to get data for {ric}. Code {dResp.status_code}, Message: {dResp.text}")
        return None
    else:
        print(f"Resource access successful for {ric}")
        jResp = json.loads(dResp.text)
        
        if isinstance(jResp, list):
            # Se la risposta è una lista, presumi che il primo elemento sia il contenuto utile
            jResp = jResp[0]

        # Converti la risposta JSON in una lista di dizionari
        data_list = []
        if "data" in jResp and "headers" in jResp:
            for d in jResp["data"]:
                record = {header["name"]: value for header, value in zip(jResp["headers"], d)}
                record["RIC"] = ric  # Aggiungi il RIC per identificare l'azienda
                data_list.append(record)
        
        return data_list
    

if __name__ == "__main__":
    # Ottieni l'ultimo token di accesso
    print("Getting OAuth access token...")
    accessToken = rdpToken.getToken()

    all_data = []

    # Itera su tutti i RIC per S&P 500
    for ric in sp500_ric_list:
        print(f"Invoking data request for {ric}")
        data = get_data_for_ric(accessToken, ric)
        if data:
            all_data.extend(data)  # Aggiungi i dati alla lista

    # Crea un DataFrame con tutti i dati raccolti
    df = pd.DataFrame(all_data)

    # # Salva il DataFrame in un file Excel
    # df.to_excel("sp500_historical_prices.xlsx", index=False)
    # Salva il DataFrame in un file CSV
    df.to_csv("sp500_historical_prices.csv", index=False)

    print("Data successfully saved to sp500_historical_prices.xlsx")
            
