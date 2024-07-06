import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
from sklearn.preprocessing import MinMaxScaler

#########################
# Verinin Okunması
#########################

df_ = pd.read_csv("flo_data_20k.csv")
df = df_.copy()
df.describe().T
df.head()
df.isnull().sum()


# Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit,0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit,0)

replace_with_thresholds(df,"order_num_total_ever_online")
replace_with_thresholds(df,"order_num_total_ever_offline")
replace_with_thresholds(df,"customer_value_total_ever_offline")
replace_with_thresholds(df,"customer_value_total_ever_online")

df["total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

df.info()
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

df["last_order_date"].max()  #Timestamp('2021-05-30 00:00:00')
analysis_date = dt.datetime(2021,6,1)


# customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
# Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.
cltv=pd.DataFrame()
cltv["customer_id"]= df["master_id"]
cltv["recency_cltv_weekly"] = ((df["last_order_date"] - df["first_order_date"]).astype("timedelta64[D]")) /7
cltv["T_weekly"] = ((analysis_date - df["first_order_date"]).astype("timedelta64[D]")) /7
cltv["frequency"] = (df["total_order"])
cltv["monetary_cltv_avg"] = df["total_value"] / df["total_order"]

# Adım1: BG/NBD modelini fit ediniz.
# • 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv
# dataframe'ine ekleyiniz.
# • 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv
# dataframe'ine ekleyiniz.

cltv.describe().T
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv['frequency'],
        cltv['recency_cltv_weekly'],
        cltv['T_weekly'])
cltv["exp_sales_3_month"] = bgf.predict(3*4,
                                        cltv["frequency"],
                                        cltv["recency_cltv_weekly"],
                                        cltv["T_weekly"])

cltv["exp_sales_6_month"] = bgf.predict(6*4,
                                        cltv["frequency"],
                                        cltv["recency_cltv_weekly"],
                                        cltv["T_weekly"])

cltv.sort_values("exp_sales_3_month",ascending=False)[:10]
cltv.sort_values("exp_sales_6_month",ascending=False)[:10]

plot_period_transactions(bgf)
plt.show()

# Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv
# dataframe'ine ekleyiniz.
# 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv['frequency'], cltv['monetary_cltv_avg'])

cltv["exp_average_value"] = ggf.conditional_expected_average_profit(cltv['frequency'],
                                        cltv['monetary_cltv_avg'])


cltv["cltv"] = ggf.customer_lifetime_value(bgf,
                                   cltv['frequency'],
                                   cltv['recency_cltv_weekly'],
                                   cltv['T_weekly'],
                                   cltv['monetary_cltv_avg'],
                                   time=6,  # aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

# en yüksek getiri sağlayacak müşteriler
cltv.sort_values("cltv",ascending=False)[:10]

## Segmentlere ayırma

cltv["cltv_segment"] = pd.qcut(cltv["cltv"],4, labels= ["D","C", "B", "A"])