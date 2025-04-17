import matplotlib.pyplot as plt
from utils import *

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

  dl_pkt_interval = dl_pkt_interval[dl_pkt_interval < 0.05] * 1000 # in ms
  ul_pkt_interval = ul_pkt_interval[ul_pkt_interval < 0.05] * 1000 # in ms

  dl_count, dl_bins_count = np.histogram(dl_pkt_interval, bins=10000) 
  dl_pdf = dl_count / sum(dl_count) 
  dl_cdf = np.cumsum(dl_pdf) 

  ul_count, ul_bins_count = np.histogram(ul_pkt_interval, bins=10000) 
  ul_pdf = ul_count / sum(ul_count) 
  ul_cdf = np.cumsum(ul_pdf) 

  # plt.figure()
  # plt.plot(dl_pkt_interval, 'r*')

  # plt.figure()
  # plt.plot(ul_pkt_interval, 'b*')

  plt.figure()
  plt.plot(dl_bins_count[1:], dl_pdf)
  plt.title("Downlink pkt inter-arrival time")

  plt.figure()
  plt.plot(ul_bins_count[1:], ul_pdf)
  plt.title("Uplink pkt inter-arrival time")

  plt.show()
  