for i in {1..10};
do
  echo Processing file ${i}...
  python3 ../src/print_ip_info.py -f ~/Documents/data/cos513/5G_Traffic_Datasets/Game_Streaming/KT_GameBox/KT_GameBox_${i}.csv
done