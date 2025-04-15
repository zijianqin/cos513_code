import numpy as np
import pandas as pd 
import argparse
import datetime

def load_csv(filename):
  df = pd.read_csv(filename, sep=',', header=0)
  return df

def stats_src_ip(df):
  set_src = list(set(df['Source']))
  print("There are {} ips in the source ip".format(len(set_src)))
  for i in range(len(set_src)):
    new_df = df['Source'][df['Source'] == set_src[i]]
    print("{} appears {} times in source ip.".format(set_src[i], new_df.shape[0]))

def stats_dst_ip(df):
  set_src = list(set(df['Destination']))
  print("There are {} ips in the dest ip".format(len(set_src)))
  for i in range(len(set_src)):
    new_df = df['Destination'][df['Destination'] == set_src[i]]
    print("{} appears {} times in dest ip.".format(set_src[i], new_df.shape[0]))

def extract_dl_pkt_interval_time(df, dst_ip):
  data = df['Time'][df['Destination'] == dst_ip]
  res = np.zeros([data.shape[0]])
  for i in range(data.shape[0]):
    res[i] = datetime.datetime.strptime(data.array[i],
        "%Y-%m-%d %H:%M:%S.%f").timestamp()
    # print(datetime.datetime.strptime(data.array[i],
    #     "%Y-%m-%d %H:%M:%S.%f").timestamp())
  return res[1:] - res[:-1]

def extract_ul_pkt_interval_time(df, src_ip):
  data = df['Time'][df['Source'] == src_ip]
  res = np.zeros([data.shape[0]])
  for i in range(data.shape[0]):
    res[i] = datetime.datetime.strptime(data.array[i],
        "%Y-%m-%d %H:%M:%S.%f").timestamp()
    # print(datetime.datetime.strptime(data.array[i],
    #     "%Y-%m-%d %H:%M:%S.%f").timestamp())
  return res[1:] - res[:-1]
  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-f', '--filename', help="directory of your data")
  parser.add_argument('-l', '--local_ip', help="local ip address")
  args = parser.parse_args()
  
  df = load_csv(args.filename)
  # stats_src_ip(df)
  # stats_dst_ip(df)
  dl_pkt_interval = extract_dl_pkt_interval_time(df, args.local_ip)
  ul_pkt_interval = extract_ul_pkt_interval_time(df, args.local_ip)
  

