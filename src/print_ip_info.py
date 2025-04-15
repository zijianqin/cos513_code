from utils import *

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-f', '--filename', help="directory of your data")
  args = parser.parse_args()
  
  df = load_csv(args.filename)
  stats_src_ip(df)
  stats_dst_ip(df)